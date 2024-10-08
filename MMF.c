#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Función para encontrar la mediana en un array
unsigned char find_median(unsigned char *window, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (window[i] > window[j]) {
                unsigned char temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }
    return window[size / 2];
}

// Función para aplicar el filtro de mediana a una sección de la imagen
void apply_mmf_section(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int window_size = 3;
    int window_half = window_size / 2;
    unsigned char window[window_size * window_size];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int count = 0;
                for (int wy = -window_half; wy <= window_half; wy++) {
                    for (int wx = -window_half; wx <= window_half; wx++) {
                        int ny = y + wy;
                        int nx = x + wx;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            window[count++] = input[(ny * width + nx) * channels + c];
                        }
                    }
                }
                output[(y * width + x) * channels + c] = find_median(window, count);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    // Comprobar los argumentos de la línea de comandos
    if (argc != 4) {
        if (rank == 0) {
            printf("Usage: %s <input_image> <output_image> <num_nodes>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int width, height, channels;
    unsigned char *image;
    // Cargar la imagen de entrada solo en el proceso 0
    if (rank == 0) {
        image = stbi_load(argv[1], &width, &height, &channels, 0);
        if (!image) {
            printf("Error loading image %s\n", argv[1]);
            MPI_Finalize();
            return 1;
        }
    }

    int num_nodes = atoi(argv[3]);   // Número de nodos (procesos) en el clúster
    int total_size = width * height * channels;
    int section_size = total_size / size;  // Calcular el tamaño de la sección para cada proceso
    unsigned char *input_section = (unsigned char *)malloc(section_size * sizeof(unsigned char));  // Sección de datos de entrada para este proceso
    unsigned char *output_section = (unsigned char *)malloc(section_size * sizeof(unsigned char));  // Sección de datos de salida para este proceso

    // Distribuir las secciones de datos a todos los procesos
    MPI_Scatter(image, section_size, MPI_UNSIGNED_CHAR, input_section, section_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Aplicar el filtro de mediana a la sección de datos localmente
    apply_mmf_section(input_section, output_section, width, height, channels);

    // Recolectar las secciones de salida de todos los procesos en el proceso 0
    MPI_Gather(output_section, section_size, MPI_UNSIGNED_CHAR, image, section_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Guardar la imagen de salida solo desde el proceso 0
    if (rank == 0) {
        if (!stbi_write_png(argv[2], width, height, channels, image, width * channels)) {
            printf("Error writing image %s\n", argv[2]);
            free(image);
            MPI_Finalize();
            return 1;
        }
        stbi_image_free(image);  // Liberar la memoria de la imagen de entrada
    }

    free(input_section);   // Liberar la memoria de la sección de datos de entrada
    free(output_section);  // Liberar la memoria de la sección de datos de salida

    MPI_Finalize();  // Finalizar MPI
    return 0;
}
