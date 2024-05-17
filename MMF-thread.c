#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Estructura para pasar parámetros a los hilos
typedef struct {
    unsigned char *input;   // Puntero a la imagen de entrada
    unsigned char *output;  // Puntero a la imagen de salida
    int width;              // Ancho de la imagen
    int height;             // Alto de la imagen
    int channels;           // Número de canales de la imagen (e.g., 3 para RGB)
    int window_size;        // Tamaño de la ventana del filtro de mediana
    int start_row;          // Fila de inicio de la sección a procesar
    int end_row;            // Fila de fin de la sección a procesar
} FilterParams;

// Función para comparar dos valores (utilizado por qsort)
int compare(const void *a, const void *b) {
    return (*(unsigned char *)a - *(unsigned char *)b);
}

// Función para aplicar el filtro de mediana a una parte de la imagen
void apply_median_filter_section(FilterParams *params) {
    int pad = params->window_size / 2;  // Mitad del tamaño de la ventana
    int window_area = params->window_size * params->window_size;
    unsigned char *window = (unsigned char *)malloc(window_area * sizeof(unsigned char));  // Array para la ventana del filtro

    // Recorrer la sección de la imagen
    for (int y = params->start_row; y < params->end_row; y++) {
        for (int x = 0; x < params->width; x++) {
            for (int c = 0; c < params->channels; c++) {
                int count = 0;
                // Recorrer los píxeles dentro de la ventana
                for (int ky = -pad; ky <= pad; ky++) {
                    for (int kx = -pad; kx <= pad; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        // Comprobar los límites de la imagen
                        if (nx >= 0 && nx < params->width && ny >= 0 && ny < params->height) {
                            window[count++] = params->input[(ny * params->width + nx) * params->channels + c];
                        }
                    }
                }
                // Ordenar los valores en la ventana y encontrar la mediana
                qsort(window, count, sizeof(unsigned char), compare);
                params->output[(y * params->width + x) * params->channels + c] = window[count / 2];
            }
        }
    }
    free(window);  // Liberar la memoria de la ventana
}

// Función que será ejecutada por cada hilo
void *filter_thread(void *arg) {
    FilterParams *params = (FilterParams *)arg;  // Convertir el argumento a un puntero a FilterParams
    apply_median_filter_section(params);         // Aplicar el filtro a la sección especificada
    return NULL;
}

// Función para dividir la imagen en secciones y crear hilos para el procesamiento
void parallel_median_filter(unsigned char *input, unsigned char *output, int width, int height, int channels, int window_size, int num_nodes) {
    pthread_t threads[num_nodes];
    FilterParams params[num_nodes];

    int rows_per_thread = height / num_nodes;  // Filas por nodo
    for (int i = 0; i < num_nodes; i++) {
        params[i].input = input;
        params[i].output = output;
        params[i].width = width;
        params[i].height = height;
        params[i].channels = channels;
        params[i].window_size = window_size;
        params[i].start_row = i * rows_per_thread;
        params[i].end_row = (i == num_nodes - 1) ? height : (i + 1) * rows_per_thread;
        
        pthread_create(&threads[i], NULL, filter_thread, &params[i]);  // Crear el hilo
    }

    // Esperar a que todos los hilos terminen
    for (int i = 0; i < num_nodes; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main(int argc, char *argv[]) {
    // Comprobar los argumentos de la línea de comandos
    if (argc != 5) {
        printf("Usage: %s <input_image> <output_image> <window_size> <num_nodes>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    // Cargar la imagen de entrada
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image %s\n", argv[1]);
        return 1;
    }

    int window_size = atoi(argv[3]);   // Tamaño de la ventana del filtro
    int num_nodes = atoi(argv[4]);     // Número de nodos
    unsigned char *output = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));  // Imagen de salida

    // Aplicar el filtro de mediana en paralelo
    parallel_median_filter(image, output, width, height, channels, window_size, num_nodes);

    // Guardar la imagen de salida
    if (!stbi_write_png(argv[2], width, height, channels, output, width * channels)) {
        printf("Error writing image %s\n", argv[2]);
        free(image);
        free(output);
        return 1;
    }

    stbi_image_free(image);  // Liberar la memoria de la imagen de entrada
    free(output);            // Liberar la memoria de la imagen de salida
    return 0;
}
