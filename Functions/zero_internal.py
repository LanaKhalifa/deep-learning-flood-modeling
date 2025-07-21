def zero_internal(matrix, boundary_thickness):
    print('ttttttttttttttttttttttttttttt', matrix.shape)
    rows, cols = matrix.shape
    for i in range(boundary_thickness, rows - boundary_thickness):
        for j in range(boundary_thickness, cols - boundary_thickness):
            matrix[i][j] = 0
    return matrix
