#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Función para aplicar el filtro DDF a una sección de la imagen
void apply_ddf_section(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int kernel_size = 3;
    int kernel_half = kernel_size / 2;
    int weights[3][3] = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                for (int ky = -kernel_half; ky <= kernel_half; ky++) {
                    for (int kx = -kernel_half; kx <= kernel_half; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                            sum += input[(ny * width + nx) * channels + c] * weights[ky + kernel_half][kx + kernel_half];
                        }
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)(sum > 255 ? 255 : (sum < 0 ? 0 : sum));
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

    // Aplicar el filtro DDF a la sección de datos localmente
    apply_ddf_section(input_section, output_section, width, height, channels);

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
