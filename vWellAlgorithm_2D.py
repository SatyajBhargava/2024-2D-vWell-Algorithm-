# Segmenting Homogeneous Regions in Images using Variance Wells (Appendix 2)
# Written by Satyaj Bhargava

# Running Python version 3.11
from scipy.stats import ttest_ind_from_stats  # running version 1.14.0
from collections import defaultdict  # default package
from heapq import heappop, heappush  # default package
import matplotlib.pyplot as plt  # running version 3.9.1
from scipy.ndimage import zoom  # running version 1.14.0
import nibabel as nib  # running version 5.2.1
import numpy as np  # running version 1.26.4
import time  # default package
import cv2  # opencv-python version 4.10.0.84

# VWELL ALGORITHM
def ManageVwellAlgorithm(input_image, kernel_radius):
    # Running and managing the vWell algorithm
    # Load or compute the variance image
    numpy_variance_image = ComputeVarianceImage(resampled_input_image, kernel_radius)

    # Find the direction image of the variance image
    orthogonal_offsets = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))
    reversed_offset_indices = (0, 3, 4, 1, 2)
    direction_image, list_of_minima = ComputeDirectionImage(numpy_variance_image, orthogonal_offsets, reversed_offset_indices)

    # Build the graph structure
    vWell_graph = vWellGraph()

    # Find vWells and fill them
    vWell_ID_image = FindVwells(direction_image, list_of_minima, vWell_graph, orthogonal_offsets, reversed_offset_indices)

    # Find vWell neighbors
    FindVwellNeighbors(vWell_ID_image, vWell_graph, orthogonal_offsets)

    # Find vWell mean image and standard deviation image
    mean_image = FindVwellMeanStdevTotal(input_image, vWell_graph)

    return mean_image, vWell_ID_image, vWell_graph, numpy_variance_image, direction_image

# To interpolate image to achieve higher resolution
def LinearlyInterpolateImage(numpy_input_image, interpolation_factor):
    # Turn itk image into a numpy array
    image_height, image_width = numpy_input_image.shape
    print(f"\tNumpy Dimensions: {image_height} x {image_width}")

    # Double sample the input image by liner interpolation
    resampled_numpy_input_image = zoom(numpy_input_image, interpolation_factor, order=1)

    return resampled_numpy_input_image

def ComputeVarianceImage(input_image, neighborhood_radius):
    print("Computing variance image...")
    start = time.perf_counter_ns()
    image_height, image_width = input_image.shape
    # Pad the image
    padded_input_image = np.pad(input_image, ((1, 1), (1, 1)), mode='edge').astype(np.uint32)

    neighborhood_width = 2 * neighborhood_radius + 1  # 3 in 2D
    neighborhood_area = neighborhood_width ** 2  # 3x3 or 9 in 2D

    # Create the variance image
    variance_image = np.zeros([image_height, image_width])

    # Compute N^2 x variance for each 3x3x3 neighborhood
    for i in range(image_height):
        for j in range(image_width):
            neighborhood = padded_input_image[i:(i + neighborhood_width), j:(j + neighborhood_width)]

            squareOfSum = np.sum(neighborhood) * np.sum(neighborhood)
            sumOfSquares = np.sum(neighborhood * neighborhood)

            variance = (neighborhood_area * sumOfSquares) - squareOfSum

            variance_image[i, j] = variance

    end = time.perf_counter_ns()
    print(f'\tVariance image calculation took: {(end - start) / 1_000_000_000:.3f} seconds.')

    return variance_image

def ComputeDirectionImage(variance_image, neighbor_offsets, pointing_at_you):
    print("Finding direction image...")
    start = time.perf_counter_ns()

    image_height, image_width = variance_image.shape
    direction_image = np.zeros([image_height, image_width], dtype=int)
    list_of_minima = []

    # Labeling each pixel
    for i in range(image_height):
        for j in range(image_width):
            # Get only the variances of the orthogonal neighbors
            last_variance = variance_image[i, j]
            for direction_you_point in range(len(neighbor_offsets)):
                # If you're checking yourself
                if direction_you_point == 0:
                    last_variance = variance_image[i, j]
                    continue

                m = i + neighbor_offsets[direction_you_point][0]
                n = j + neighbor_offsets[direction_you_point][1]

                # Make sure neighbor is within bounds
                if 0 <= m < image_height and 0 <= n < image_width:
                    # If neighbor is pointing at you, continue
                    if direction_image[m, n] == pointing_at_you[direction_you_point]:
                        continue  # To next neighbor

                    # Point at the smallest variance within neighbors
                    elif variance_image[m, n] <= last_variance:
                        direction_image[i, j] = direction_you_point
                        last_variance = variance_image[m, n]

                    # If the variances are equal
                    elif variance_image[m, n] == last_variance:
                        continue  # Enforce priority

            # If the code reaches here, direction_image[i, j] == 0
            if direction_image[i, j] == 0:
                list_of_minima.append([i, j])

    end = time.perf_counter_ns()
    print(f"\tLocal minima found = {np.sum(direction_image == 0)}")
    print(f'\tFinding direction image took: {(end - start) / 1_000_000_000:.3f} seconds.')

    return direction_image, list_of_minima

# Retracing the pixel graph structure from direction image
def FindVwells(direction_image, list_of_minima, vWell_graph, neighbor_offsets, pointing_at_you):
    print("Finding vWells...")
    start = time.perf_counter_ns()

    image_height, image_width = direction_image.shape
    vWell_ID_image = np.zeros([image_height, image_width], dtype=int)
    total_local_minima = len(list_of_minima)

    EMPTY_PIXEL = 0
    for minima in range(total_local_minima):
        i, j = list_of_minima[minima]
        vector_of_pixels_in_vWell = {(i, j)}  # Set
        vector_stack_floodfill = {(i, j)}  # Set
        vWell_graph.vWell_roots[minima] = (i, j)  # Tuple

        while vector_stack_floodfill:
            a, b = vector_stack_floodfill.pop()

            # If current pixel is not filled, fill
            if vWell_ID_image[a, b] == EMPTY_PIXEL:
                vWell_ID_image[a, b] = minima
            else:  # Already filled
                continue

            for direction_you_point in range(len(neighbor_offsets)):
                # If you're checking yourself
                if direction_you_point == 0:
                    continue

                m = a + neighbor_offsets[direction_you_point][0]
                n = b + neighbor_offsets[direction_you_point][1]

                # Make sure neighbor is within bounds
                if 0 <= m < image_height and 0 <= n < image_width:
                    # If neighbor is pointing at you, add to floodfill stack
                    if direction_image[m, n] == pointing_at_you[direction_you_point]:
                        vector_stack_floodfill.add((m, n))
                        vector_of_pixels_in_vWell.add((m, n))

        vWell_graph.vWell_pixel_locations[minima].update(vector_of_pixels_in_vWell)

    end = time.perf_counter_ns()
    print(f'\tFloodfilling took: {(end - start) / 1_000_000_000:.3f} seconds.')

    return vWell_ID_image

def FindVwellNeighbors(vWell_ID_image, vWell_graph, neighbor_offsets):
    print("Finding vWell neighbors...")
    start = time.perf_counter_ns()

    image_height, image_width = vWell_ID_image.shape

    # Go through each pixel and record neighbors
    for i in range(image_height):
        for j in range(image_width):
            your_ID = vWell_ID_image[i, j]

            for neighbor in range(len(neighbor_offsets)):
                m = i + neighbor_offsets[neighbor][0]
                n = j + neighbor_offsets[neighbor][1]

                # Make sure neighbor is within bounds
                if 0 <= m < image_height and 0 <= n < image_width:
                    neighbor_ID = vWell_ID_image[m, n]
                    if neighbor_ID != your_ID:
                        vWell_graph.AddEdge(neighbor_ID, your_ID)

    end = time.perf_counter_ns()
    print(f'\tFinding vWell neighbors took: {(end - start) / 1_000_000_000:.3f} seconds.')

def FindVwellMeanStdevTotal(input_image, vWell_graph):
    print(f"Finding means and standard deviations...")
    start = time.perf_counter_ns()

    # An image that shows the means of each vWell region
    image_height, image_width = input_image.shape
    vWell_mean_image = np.zeros([image_height, image_width])

    # For each vWell
    for ID, vWell in vWell_graph.vWell_pixel_locations.items():
        # Calculate the mean and standard deviation for each vWell
        values = [input_image[i, j] for i, j in vWell]
        mean = np.mean(values)
        stdev = np.std(values)
        total_n = len(values)

        vWell_graph.vWell_means[ID] = mean
        vWell_graph.vWell_stdevs[ID] = stdev
        vWell_graph.vWell_total_pixels[ID] = total_n

        for i, j in vWell:
            vWell_mean_image[i, j] = mean

    end = time.perf_counter_ns()
    print(f'\tFinding means and stdevs neighbors took: {(end - start) / 1_000_000_000:.3f} seconds.')

    return vWell_mean_image

# Manage the Dijkstra's algorithm for the selected points
def FindShortestPath(vWell_ID_image, vWell_graph, points, aim_intensity, resampling_factor):
    print("Finding shortest path...")
    start = time.perf_counter_ns()

    shortest_paths = []

    for n in range(len(points) - 1):
        # Find the starting and ending vWells (nodes)
        j, i = (points[n][0] * resampling_factor, points[n][1] * resampling_factor)
        start_vWell = int(vWell_ID_image[i, j])
        j, i = (points[n + 1][0] * resampling_factor, points[n + 1][1] * resampling_factor)
        end_vWell = int(vWell_ID_image[i, j])

        # Get the shortest paths between the points
        individual_shortest_path = vWell_graph.DijkstraShortestPath(start_vWell, end_vWell, aim_intensity)

        # Add the shortest path to the list of paths
        shortest_paths.append(individual_shortest_path)

    full_shortest_path = [vWell for path in shortest_paths for vWell in path]
    print(f"\tvWells in shortest path: {len(full_shortest_path)}")

    end = time.perf_counter_ns()
    print(f'\tTotal time taken for path finding was {(end - start) / 1_000_000_000:.3f} seconds.')

    return full_shortest_path

def RegionGrowFromPath(shortest_path, vWell_graph):
    print("Region growing...")
    start = time.perf_counter_ns()

    # Make a set for all the vWells in the shortest path (blob)
    blob = set(shortest_path)

    # Make a set of all the neighbors of the blob (background)
    background = set()
    for vWell in blob:
        background.update(vWell_graph.NeighborsOf(vWell))
    # Remove all the vWells that are in the blob
    background.difference_update(blob)
    print(f"\tOriginal vWells in blob: {len(blob)}")
    print(f"\tOriginal vWells in background: {len(background)}")

    # Start the region growing
    while True:
        # Compare current blob and background
        anchor_t_value = vWell_graph.CompareBlobAndBackground(blob, background)

        # Get the best vWell from the current background
        vWell = GetBestBackgroundVwell(vWell_graph, blob, background)

        # Transfer the vWell from background to blob
        if vWell not in blob:
            blob.add(vWell)
        if vWell in background:
            background.remove(vWell)

        # Get neighbors of vWell
        neighbor_vWells = vWell_graph.NeighborsOf(vWell)
        # Add vWells that aren't in blob to the temporary background
        neighbor_vWells.difference_update(blob)
        background.update(neighbor_vWells)

        # Compare new background and blob
        new_t_value = vWell_graph.CompareBlobAndBackground(blob, background)

        if new_t_value < anchor_t_value:
            break

    end = time.perf_counter_ns()
    print(f'\tNew length of blob: {len(blob)}')
    print(f'\tNew length of background: {len(background)}')
    print(f'\tTime taken for region growing was {(end - start) / 1_000_000_000:.3f} seconds.')

    return blob, background

# Find the vWell that increases the t-value metric the most
def GetBestBackgroundVwell(vWell_graph, blob, background):
    # Sort the vector by which vWell is best compared to the current background and blob
    stack_heap = []  # min heap to hold indices of local minima
    for vWell in background:
        # Create a copy of the background and blob
        temporary_blob = blob.copy()
        temporary_background = background.copy()

        # Transfer the vWell from background to blob
        if vWell not in temporary_blob:
            temporary_blob.add(vWell)
        if vWell in temporary_background:
            temporary_background.remove(vWell)

        # Get neighbors of vWell
        neighbor_vWells = vWell_graph.NeighborsOf(vWell)
        # Add vWells that aren't in blob to the temporary background
        neighbor_vWells.difference_update(temporary_blob)
        temporary_background.update(neighbor_vWells)

        # Get t value of new blob and background
        t_value = vWell_graph.CompareBlobAndBackground(temporary_blob, temporary_background)

        # Add the negative t value and vWell to the heap
        # So that when we heappop, we get the largest vWell
        heappush(stack_heap, (-1 * t_value, vWell))

    t_value, best_vWell = heappop(stack_heap)

    return best_vWell

# Class for the vWell graph structure and all the associated graph operations
class vWellGraph:
    def __init__(self):
        self.graph = defaultdict(set)
        self.vWell_means = defaultdict(np.float64)
        self.vWell_stdevs = defaultdict(np.float64)
        self.vWell_total_pixels = defaultdict(int)
        self.vWell_roots = defaultdict(tuple)
        self.vWell_pixel_locations = defaultdict(set)

    def AddEdge(self, node1, node2):
        self.graph[node1].add(node2)
        self.graph[node2].add(node1)

    def NeighborsOf(self, node):
        return self.graph.get(node).copy()

    # Shortest path
    def CalculateDistance(self, node1, node2):
        # distance is the t_value between two nodes
        mean1 = self.vWell_means.get(node1)
        mean2 = self.vWell_means.get(node2)

        stdev1 = self.vWell_stdevs.get(node1)
        stdev2 = self.vWell_stdevs.get(node2)

        n1 = self.vWell_total_pixels.get(node1)
        n2 = self.vWell_total_pixels.get(node2)

        t_value, _ = ttest_ind_from_stats(mean1=mean1, std1=stdev1, nobs1=n1, mean2=mean2, std2=stdev2, nobs2=n2)
        distance = abs(t_value)

        return distance

    def CalculateDirectionalWeight(self, current_node, neighbor, end_node):
        # Get positions of the nodes
        current_position = np.array(self.vWell_roots.get(current_node))
        neighbor_position = np.array(self.vWell_roots.get(neighbor))
        end_position = np.array(self.vWell_roots.get(end_node))

        # Calculate direction vectors
        current_to_end = end_position - current_position
        neighbor_to_end = end_position - neighbor_position

        # Calculate distance
        current_to_end_distance = np.sqrt(np.sum(current_to_end ** 2))
        neighbor_to_end_distance = np.sqrt(np.sum(neighbor_to_end ** 2))

        # Calculate weight
        directional_weight = neighbor_to_end_distance / current_to_end_distance

        return directional_weight

    def DijkstraShortestPath(self, start_node, end_node, directional_bias):
        # Initialize data structures
        min_heap = [(0, start_node)]
        shortest_distances = {node: float('inf') for node in self.graph}
        shortest_distances[start_node] = 0
        last_nodes = defaultdict(int)
        visited_nodes = set()

        while min_heap:
            current_distance, current_node = heappop(min_heap)

            if current_node == end_node:
                break

            if current_distance > shortest_distances.get(current_node):
                continue

            for neighbor in self.NeighborsOf(current_node):
                if neighbor in visited_nodes:
                    continue

                # Calculate t-test weight
                t_test_weight = self.CalculateDistance(current_node, neighbor)

                # Calculate directional weight
                directional_weight = self.CalculateDirectionalWeight(current_node, neighbor, end_node)
                # Apply directional weight to the distance
                weighted_distance = (directional_bias * directional_weight) + t_test_weight

                total_distance = current_distance + weighted_distance

                if total_distance < shortest_distances.get(neighbor):
                    shortest_distances[neighbor] = total_distance
                    last_nodes[neighbor] = current_node
                    heappush(min_heap, (total_distance, neighbor))

            visited_nodes.add(current_node)

        if shortest_distances[end_node] == float('inf'):
            print(f"\tNo path found")
            return None  # No path found

        # Reconstruct path
        shortest_path = []
        current = end_node
        while current != start_node:
            shortest_path.append(current)
            current = last_nodes[current]
        shortest_path.append(start_node)
        shortest_path.reverse()

        return shortest_path

    def CompareBlobAndBackground(self, blob, background):
        # Cache values
        vWell_means_blob = np.array([self.vWell_means.get(vWell) for vWell in blob])
        vWell_pixels_blob = np.array([self.vWell_total_pixels.get(vWell) for vWell in blob])

        vWell_means_background = np.array([self.vWell_means.get(vWell) for vWell in background])
        vWell_pixels_background = np.array([self.vWell_total_pixels.get(vWell) for vWell in background])

        # Blob calculations
        blob_sum = np.sum(vWell_means_blob * vWell_pixels_blob)
        blob_total_n = np.sum(vWell_pixels_blob)
        blob_mean = blob_sum / blob_total_n
        # Weighted variance calculation for stdev
        blob_variance = np.sum(vWell_pixels_blob * (vWell_means_blob - blob_mean) ** 2) / blob_total_n
        blob_stdev = np.sqrt(blob_variance)

        # Background calculations
        background_sum = np.sum(vWell_means_background * vWell_pixels_background)
        background_total_n = np.sum(vWell_pixels_background)
        background_mean = background_sum / background_total_n
        background_variance = np.sum(vWell_pixels_background * (vWell_means_background - background_mean) ** 2) / background_total_n
        background_stdev = np.sqrt(background_variance)

        # T-test
        signed_t, _ = ttest_ind_from_stats(mean1=blob_mean, std1=blob_stdev, nobs1=blob_total_n, mean2=background_mean, std2=background_stdev, nobs2=background_total_n, equal_var=False)
        test_statistic = abs(signed_t)

        return test_statistic

# MISCELLANEOUS FUNCTIONS
def GetShortestPathLine(input_image, vWell_graph, vWells_in_shortest_path):
    # An image that shows the means of each vWell region
    image_height, image_width = input_image.shape
    binary_shortest_path_image = np.zeros([image_height, image_width], dtype=np.uint8)

    # List of points (x, y) coordinates
    shortest_path_local_minima = [vWell_graph.vWell_roots[int(vWell)] for vWell in vWells_in_shortest_path]

    # Draw lines between the points on the mask
    for i in range(len(shortest_path_local_minima) - 1):
        start_point = (shortest_path_local_minima[i][1], shortest_path_local_minima[i][0])
        end_point = (shortest_path_local_minima[i + 1][1], shortest_path_local_minima[i + 1][0])
        cv2.line(binary_shortest_path_image, start_point, end_point, color=255, thickness=1)

    highlighted_shortest_path_image = MakeWhiteRed(binary_shortest_path_image, input_image)

    return highlighted_shortest_path_image

def GetFilledRegion(input_image, vWell_graph, vWells_in_region):
    # An image that shows the means of each vWell region
    image_height, image_width = input_image.shape
    binary_region_image = np.zeros([image_height, image_width])

    for vWell in vWells_in_region:
        for i, j in vWell_graph.vWell_pixel_locations.get(vWell):
            binary_region_image[i, j] = 1

    highlighted_shortest_path_image = MakeWhiteRed(binary_region_image, input_image)

    return highlighted_shortest_path_image

def MakeWhiteRed(to_be_red_image, overlayed_on_image):
    # Create a copy of the segmented image to avoid modifying the original
    highlighted_image = overlayed_on_image.copy()

    # Thresholding for white pixels
    white_mask = to_be_red_image > 0

    # Make the two RGB Images
    highlighted_opaque_image = cv2.cvtColor(highlighted_image.astype(np.float32), cv2.COLOR_GRAY2RGB)

    # Set the corresponding pixels to red in the highlighted image
    highlighted_opaque_image[white_mask] = [255, 0, 0]

    return highlighted_opaque_image

def NormalizeImage(input_image):
    # Normalize the image to 0-255 range for proper display
    normalized_image = cv2.normalize(input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_image

def ShowImages(**Images):
    print("Showing Images...\n")

    # Set up the number of images in a single row
    num_images = len(Images)
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(12, 5))

    for (name, image), ax in zip(Images.items(), axes.flatten()):
        # Scale image to [0, 255] if necessary
        scaled_image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

        ax.imshow(scaled_image, cmap='gray')
        ax.set_title(name)
        ax.axis('off')  # Turn off axis numbers and ticks

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Settings
    kernel_radius = 1
    resampling_factor = 2
    heuristic_weight = 0.4

    print(f"Running vWell algorithm at resampling factor: {resampling_factor}")
    final_start = time.perf_counter_ns()

    # Load the image in as a NumPy array
    img = nib.load("X216DoubleCroppedSlice.nii")
    loaded_image = img.get_fdata()

    # Resample image according to resampling factor
    resampled_input_image = LinearlyInterpolateImage(loaded_image, resampling_factor)

    # Compute the vWells
    vWell_mean_image, vWell_index_image, graph_of_vWells, vWell_variance_image, direction_image = ManageVwellAlgorithm(resampled_input_image, kernel_radius)

    # Pick points on input to connect and fill from
    picked_points = [(22, 8), (32, 21), (14, 59)]

    # # Find the shortest path between selected points
    shortest_path = FindShortestPath(vWell_index_image, graph_of_vWells, picked_points, heuristic_weight, resampling_factor)

    # # Region grow from the shortest path
    vWells_in_path, vWells_around_path = RegionGrowFromPath(shortest_path, graph_of_vWells)

    # Get the images to display
    normalized_input_image = NormalizeImage(resampled_input_image)
    shortest_path_image = GetShortestPathLine(normalized_input_image, graph_of_vWells, shortest_path)
    region_filled_image = GetFilledRegion(normalized_input_image, graph_of_vWells, vWells_in_path)
    background_of_region_image = GetFilledRegion(normalized_input_image, graph_of_vWells, vWells_around_path)

    final_end = time.perf_counter_ns()
    print(f'\nTotal time taken for vWell algorithm was {(final_end - final_start) / 1_000_000_000:.3f} seconds.\n')

    ShowImages(Input=resampled_input_image, vWell_mean=vWell_mean_image, Shortest_Path=shortest_path_image, Region_Grown=region_filled_image, Background=background_of_region_image)
