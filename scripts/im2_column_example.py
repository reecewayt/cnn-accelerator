import numpy as np


def direct_convolution(input_image, kernel, stride=1):
    """Perform direct convolution by sliding the kernel over the input image."""
    input_h, input_w = input_image.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate output dimensions
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    output = np.zeros((output_h, output_w))

    # Slide kernel over input image
    for i in range(output_h):
        for j in range(output_w):
            # Extract region of interest
            roi = input_image[
                i * stride : i * stride + kernel_h, j * stride : j * stride + kernel_w
            ]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(roi * kernel)

    return output


def im2col(input_image, kernel_size, stride=1):
    """Transform image regions into columns for efficient convolution."""
    input_h, input_w = input_image.shape
    kernel_h, kernel_w = kernel_size

    # Calculate output dimensions
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    # Initialize im2col matrix
    im2col_matrix = np.zeros((kernel_h * kernel_w, output_h * output_w))

    # Fill im2col matrix
    col_idx = 0
    for i in range(0, input_h - kernel_h + 1, stride):
        for j in range(0, input_w - kernel_w + 1, stride):
            # Extract patch
            patch = input_image[i : i + kernel_h, j : j + kernel_w]
            # Flatten patch to column and insert into im2col matrix
            im2col_matrix[:, col_idx] = patch.flatten()
            col_idx += 1

    return im2col_matrix


import numpy as np


def im2col_general(input_data, kernel_size, stride=(1, 1), padding=(0, 0)):
    """
    Transform image regions into columns for efficient convolution.

    Args:
        input_data: Input tensor of shape (C, H, W) or (H, W) for single channel
        kernel_size: Tuple of (kernel_height, kernel_width)
        stride: Tuple of (stride_height, stride_width)
        padding: Tuple of (padding_height, padding_width)

    Returns:
        im2col matrix of shape (C*kernel_height*kernel_width, output_height*output_width)
    """
    # Add channel dimension if needed
    if len(input_data.shape) == 2:
        input_data = input_data.reshape(1, *input_data.shape)

    # Extract dimensions
    C, H, W = input_data.shape
    K_h, K_w = kernel_size
    S_h, S_w = stride
    P_h, P_w = padding

    # Calculate output dimensions
    out_h = (H + 2 * P_h - K_h) // S_h + 1
    out_w = (W + 2 * P_w - K_w) // S_w + 1

    # Apply padding if needed
    if P_h > 0 or P_w > 0:
        padded_data = np.pad(input_data, ((0, 0), (P_h, P_h), (P_w, P_w)), "constant")
    else:
        padded_data = input_data

    # Initialize im2col matrix
    im2col_matrix = np.zeros((C * K_h * K_w, out_h * out_w))

    # Fill im2col matrix
    for c in range(C):  # For each channel
        for k_h in range(K_h):  # For each kernel row
            for k_w in range(K_w):  # For each kernel column
                # Calculate row index in im2col matrix
                row_idx = c * K_h * K_w + k_h * K_w + k_w

                # Fill this row with appropriate values from input
                col_idx = 0
                for i in range(out_h):  # For each output row
                    for j in range(out_w):  # For each output column
                        # Calculate position in padded input
                        in_i = i * S_h + k_h
                        in_j = j * S_w + k_w

                        # Store value in im2col matrix
                        im2col_matrix[row_idx, col_idx] = padded_data[c, in_i, in_j]
                        col_idx += 1

    return im2col_matrix


def convolution_im2col(input_image, kernel, stride=1):
    """Perform convolution using im2col transformation."""
    kernel_h, kernel_w = kernel.shape

    # Get im2col matrix
    im2col_matrix = im2col(input_image, (kernel_h, kernel_w), stride)

    # Reshape kernel to row vector
    kernel_vec = kernel.flatten()

    # Matrix multiply
    output_vec = np.dot(kernel_vec, im2col_matrix)

    # Reshape output to proper dimensions
    output_h = (input_image.shape[0] - kernel_h) // stride + 1
    output_w = (input_image.shape[1] - kernel_w) // stride + 1
    output = output_vec.reshape(output_h, output_w)

    return output


def tiled_convolution_im2col(input_image, kernel, tile_size, stride=1):
    """Perform convolution using im2col with tiling for large images."""
    input_h, input_w = input_image.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate output dimensions
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    # Initialize output array
    output = np.zeros((output_h, output_w))

    # Calculate effective tile size (need overlap for kernels at the edges)
    effective_tile_h = tile_size
    effective_tile_w = tile_size

    # For each tile
    for tile_row in range(0, input_h, effective_tile_h):
        for tile_col in range(0, input_w, effective_tile_w):
            # Calculate tile boundaries (handle boundary cases)
            tile_h_start = tile_row
            tile_h_end = min(tile_row + effective_tile_h + kernel_h - 1, input_h)
            tile_w_start = tile_col
            tile_w_end = min(tile_col + effective_tile_w + kernel_w - 1, input_w)

            # Extract tile
            tile = input_image[tile_h_start:tile_h_end, tile_w_start:tile_w_end]

            # Skip tiles that are too small for the kernel
            if tile.shape[0] < kernel_h or tile.shape[1] < kernel_w:
                continue

            # Process tile with im2col
            tile_output = convolution_im2col(tile, kernel, stride)

            # Calculate where this tile's output should go in the final output
            out_h_start = max(0, (tile_h_start - kernel_h + 1) // stride)
            out_w_start = max(0, (tile_w_start - kernel_w + 1) // stride)
            out_h_end = min(output_h, (tile_h_end - kernel_h + 1) // stride)
            out_w_end = min(output_w, (tile_w_end - kernel_w + 1) // stride)

            # Copy tile output to appropriate location in final output
            output_tile_h = out_h_end - out_h_start
            output_tile_w = out_w_end - out_w_start
            output[out_h_start:out_h_end, out_w_start:out_w_end] = tile_output[
                :output_tile_h, :output_tile_w
            ]

    return output


# Example from our walkthrough
def main():
    # Create 4x4 input image
    small_input = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    # Create 3x3 kernel
    kernel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

    print("Input image:")
    print(small_input)
    print("\nKernel:")
    print(kernel)

    # Direct convolution
    direct_result = direct_convolution(small_input, kernel)
    print("\nResult using direct convolution:")
    print(direct_result)

    # Im2col convolution
    im2col_result = convolution_im2col(small_input, kernel)
    print("\nResult using im2col convolution:")
    print(im2col_result)

    # Im2col matrix (for visualization)
    im2col_matrix = im2col(small_input, kernel.shape)
    print("\nIm2col matrix:")
    print(im2col_matrix)

    # Now let's create a larger example with tiling (16x16 input)
    large_input = np.arange(1, 257).reshape(16, 16)
    print("\nLarge input (16x16):")
    print(large_input)

    # Process with tiling (simulate 16x16 PE array by using tile_size=14)
    # We use 14 because that's the output size we get from a 16x16 input with a 3x3 kernel
    tiled_result = tiled_convolution_im2col(large_input, kernel, tile_size=14)
    print("\nResult using tiled im2col convolution (first few rows):")
    print(tiled_result[:5, :5])  # Just show a portion of the output

    # Compare with direct result to verify correctness
    direct_large_result = direct_convolution(large_input, kernel)
    print("\nDirect convolution result (first few rows):")
    print(direct_large_result[:5, :5])

    # Verify that results match
    if np.allclose(tiled_result, direct_large_result):
        print("\nTiled and direct results match! ✓")
    else:
        print("\nError: Tiled and direct results do not match!")


if __name__ == "__main__":
    main()
