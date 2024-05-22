#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <mpi.h>
using namespace cv;
using namespace std;

/*
mpic++ -o DDF DDF.cpp `pkg-config --cflags --libs opencv4`
mpirun -np 4 ./DDF test-soft.png soft-output-test.png 10 50.0
*/

// Function to apply a directional diffusion filter on a part of the image
void directional_diffusion_filter_part(Mat& image_part, int iterations, double lambda) {
    Mat grad_x, grad_y, grad_mag, diffusion;
    int rows = image_part.rows;
    int cols = image_part.cols;

    for (int it = 0; it < iterations; ++it) {
        // Compute gradients
        Sobel(image_part, grad_x, CV_64F, 1, 0, 3);
        Sobel(image_part, grad_y, CV_64F, 0, 1, 3);

        // Compute gradient magnitude
        magnitude(grad_x, grad_y, grad_mag);

        // Compute diffusion coefficient
        diffusion = 1.0 / (1.0 + grad_mag);

        // Update the image based on diffusion
        for (int r = 1; r < rows - 1; ++r) {
            for (int c = 1; c < cols - 1; ++c) {
                double diff = diffusion.at<double>(r, c);
                image_part.at<uchar>(r, c) += lambda * diff * (
                    image_part.at<uchar>(r + 1, c) +
                    image_part.at<uchar>(r - 1, c) +
                    image_part.at<uchar>(r, c + 1) +
                    image_part.at<uchar>(r, c - 1) -
                    4 * image_part.at<uchar>(r, c)
                );
            }
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <input_image_path> <output_image_path> <iterations> <lambda>" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    string input_image_path = argv[1];
    string output_image_path = argv[2];
    int iterations = stoi(argv[3]);
    double lambda = stod(argv[4]);

    Mat image;
    int rows_per_node;
    int total_rows, total_cols;

    if (rank == 0) {
        // Master node loads the image
        image = imread(input_image_path, IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Error: could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // Distribute the workload
        total_rows = image.rows;
        total_cols = image.cols;
        rows_per_node = total_rows / size;

        // Send the size of the divisions to the other nodes
        for (int i = 1; i < size; ++i) {
            MPI_Send(&total_rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&total_cols, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_per_node, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Other nodes receive the size of the divisions
        MPI_Recv(&total_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&total_cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows_per_node, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Adjust the last part if it does not divide exactly
    int extra_rows = total_rows % size;
    if (rank == size - 1) {
        rows_per_node += extra_rows;
    }

    // Create the matrix for the part of the image that each node will process
    Mat image_part(rows_per_node, total_cols, CV_8UC1);
    Mat result_part;

    if (rank == 0) {
        // Master node sends parts of the image to the other nodes
        for (int i = 1; i < size; ++i) {
            int start = i * rows_per_node - (i > 0 ? extra_rows : 0);
            int rows = (i == size - 1) ? (rows_per_node + extra_rows) : rows_per_node;
            MPI_Send(image.ptr<uchar>(start), rows * total_cols, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
        }

        // Master node processes its own part
        image_part = image.rowRange(0, rows_per_node).clone();
        directional_diffusion_filter_part(image_part, iterations, lambda);
        result_part = image_part;
    } else {
        // Other nodes receive their part of the image
        MPI_Recv(image_part.data, rows_per_node * total_cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Process their part
        directional_diffusion_filter_part(image_part, iterations, lambda);
        result_part = image_part;
    }

    if (rank == 0) {
        // Master node receives the processed parts from the other nodes and concatenates them
        Mat filtered_image;
        vector<Mat> filtered_parts(size);
        filtered_parts[0] = result_part;

        for (int i = 1; i < size; ++i) {
            int rows = (i == size - 1) ? (rows_per_node + extra_rows) : rows_per_node;
            Mat part(rows, total_cols, CV_8UC1);
            MPI_Recv(part.data, rows * total_cols, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            filtered_parts[i] = part;
        }

        vconcat(filtered_parts, filtered_image);

        // Save the resulting image
        imwrite(output_image_path, filtered_image);
    } else {
        // Other nodes send their processed part to the master node
        MPI_Send(result_part.data, rows_per_node * total_cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

