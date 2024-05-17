#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
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
    int start_row;          // Fila de inicio de la sección a procesar
    int end_row;            // Fila de fin de la sección a procesar
    int iterations;         // Número de iteraciones del filtro DDF
    float lambda;           // Parámetro lambda para el filtro DDF
} FilterParams;

// Función para calcular la conductancia
float conductance(float gradient, float lambda) {
    return expf(- (gradient * gradient) / (lambda * lambda));
}

// Función para aplicar el filtro de difusión direccional a una parte de la imagen
void apply_ddf_section(FilterParams *params) {
    int width = params->width;
    int height = params->height;
    int channels = params->channels;
    int start_row = params->start_row;
    int end_row = params->end_row;
    int iterations = params->iterations;
    float lambda = params->lambda;

    // Buffer temporal para la imagen procesada en cada iteración
    unsigned char *temp = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
    memcpy(temp, params->input, width * height * channels * sizeof(unsigned char));

    // Iteraciones del filtro de difusión direccional
    for (int iter = 0; iter < iterations; iter++) {
        // Procesar cada píxel de la sección correspondiente
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    int idx = (y * width + x) * channels + c;
                    int up = ((y - 1) * width + x) * channels + c;
                    int down = ((y + 1) * width + x) * channels + c;
                    int left = (y * width + (x - 1)) * channels + c;
                    int right = (y * width + (x + 1)) * channels + c;

                    // Calcular las diferencias de intensidad con los píxeles vecinos
                    float deltaN = (y > 0) ? (temp[up] - temp[idx]) : 0.0f;
                    float deltaS = (y < height - 1) ? (temp[down] - temp[idx]) : 0.0f;
                    float deltaE = (x < width - 1) ? (temp[right] - temp[idx]) : 0.0f;
                    float deltaW = (x > 0) ? (temp[left] - temp[idx]) : 0.0f;

                    // Calcular los coeficientes de conductancia
                    float cN = conductance(deltaN, lambda);
                    float cS = conductance(deltaS, lambda);
                    float cE = conductance(deltaE, lambda);
                    float cW = conductance(deltaW, lambda);

                    // Actualizar el valor del píxel aplicando el filtro DDF
                    params->output[idx] = temp[idx] + 0.25 * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
                }
            }
        }
        // Copiar la salida a la entrada para la próxima iteración
        memcpy(temp + start_row * width * channels, params->output + start_row * width * channels, (end_row - start_row) * width * channels);
    }

    free(temp);
}

// Función que será ejecutada por cada hilo
void *filter_thread(void *arg) {
    FilterParams *params = (FilterParams *)arg;  // Convertir el argumento a un puntero a FilterParams
    apply_ddf_section(params);                   // Aplicar el filtro DDF a la sección especificada
    return NULL;
}

// Función para dividir la imagen en secciones y crear hilos para el procesamiento
void parallel_ddf_filter(unsigned char *input, unsigned char *output, int width, int height, int channels, int iterations, float lambda, int num_nodes) {
    pthread_t threads[num_nodes];  // Array para almacenar los identificadores de los hilos
    FilterParams params[num_nodes]; // Array para almacenar los parámetros de cada hilo

    int rows_per_thread = height / num_nodes;  // Calcular el número de filas por hilo
    for (int i = 0; i < num_nodes; i++) {
        params[i].input = input;
        params[i].output = output;
        params[i].width = width;
        params[i].height = height;
        params[i].channels = channels;
        params[i].iterations = iterations;
        params[i].lambda = lambda;
        params[i].start_row = i * rows_per_thread;  // Fila de inicio para este hilo
        params[i].end_row = (i == num_nodes - 1) ? height : (i + 1) * rows_per_thread;  // Fila de fin para este hilo
        
        pthread_create(&threads[i], NULL, filter_thread, &params[i]);  // Crear el hilo
    }

    // Esperar a que todos los hilos terminen
    for (int i = 0; i < num_nodes; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main(int argc, char *argv[]) {
    // Comprobar los argumentos de la línea de comandos
    if (argc != 6) {
        printf("Usage: %s <input_image> <output_image> <iterations> <lambda> <num_nodes>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    // Cargar la imagen de entrada
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image %s\n", argv[1]);
        return 1;
    }

    int iterations = atoi(argv[3]);  // Número de iteraciones del filtro DDF
    float lambda = atof(argv[4]);    // Parámetro lambda para el filtro DDF
    int num_nodes = atoi(argv[5]);   // Número de nodos (hilos) para el procesamiento paralelo
    unsigned char *output = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));  // Imagen de salida

    // Aplicar el filtro de difusión direccional en paralelo
    parallel_ddf_filter(image, output, width, height, channels, iterations, lambda, num_nodes);

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
