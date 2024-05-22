#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <mpi.h>
using namespace cv;
using namespace std;

/*
mpic++ -o MMF MMF.cpp `pkg-config --cflags --libs opencv4`
mpirun -np 4 ./MMF test-noise.png noise-output-test.png 5
*/

// Function to apply a median filter on a part of the image for each channel
void median_filter_part(const Mat& image_part, int filter_size, Mat& result) {
    vector<Mat> channels(3);
    split(image_part, channels);
    for (int i = 0; i < 3; ++i) {
        medianBlur(channels[i], channels[i], filter_size);
    }
    merge(channels, result);
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <input_image_path> <output_image_path> <filter_size>" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    string input_image_path = argv[1];
    string output_image_path = argv[2];
    int filter_size = stoi(argv[3]);

    Mat image;
    int rows_per_node;
    int total_rows, total_cols;

    if (rank == 0) {
        // Master node loads the image
        image = imread(input_image_path, IMREAD_COLOR);
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
    Mat image_part(rows_per_node, total_cols, CV_8UC3);
    Mat result_part;

    if (rank == 0) {
        // Master node sends parts of the image to the other nodes
        for (int i = 1; i < size; ++i) {
            int start = i * rows_per_node - (i > 0 ? extra_rows : 0);
            int rows = (i == size - 1) ? (rows_per_node + extra_rows) : rows_per_node;
            MPI_Send(image.ptr<uchar>(start), rows * total_cols * 3, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
        }

        // Master node processes its own part
        image_part = image.rowRange(0, rows_per_node).clone();
        median_filter_part(image_part, filter_size, result_part);
    } else {
        // Other nodes receive their part of the image
        MPI_Recv(image_part.data, rows_per_node * total_cols * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Process their part
        median_filter_part(image_part, filter_size, result_part);
    }

    if (rank == 0) {
        // Master node receives the processed parts from the other nodes and concatenates them
        Mat filtered_image;
        vector<Mat> filtered_parts(size);
        filtered_parts[0] = result_part;

        for (int i = 1; i < size; ++i) {
            int rows = (i == size - 1) ? (rows_per_node + extra_rows) : rows_per_node;
            Mat part(rows, total_cols, CV_8UC3);
            MPI_Recv(part.data, rows * total_cols * 3, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            filtered_parts[i] = part;
        }

        vconcat(filtered_parts, filtered_image);

        // Save the resulting image
        imwrite(output_image_path, filtered_image);
    } else {
        // Other nodes send their processed part to the master node
        MPI_Send(result_part.data, rows_per_node * total_cols * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

