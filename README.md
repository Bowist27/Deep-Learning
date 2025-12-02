# Deep-Learning
Arquitectura de deep learning para solucionar problema comun de kaggle (Imágenes)

## Ejecución en Google Colab con Docker (Runtime Local)

Para conectar Colab a tu contenedor Docker local:

1.  **Correr el Contenedor:**
    Ejecuta tu imagen de Docker asegurándote de exponer el puerto y permitir el origen de Colab. Usa este comando (ajusta `nombre_imagen`):
    ```bash
    docker run -p 8888:8888 --rm -it nombre_imagen jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0 --ip=0.0.0.0 --allow-root
    ```

2.  **Conectar en Colab:**
    *   Copia la URL con el token que aparece en la terminal (ej. `http://127.0.0.1:8888/?token=...`).
    *   En Colab, ve a **Conectar** > **Conectar a un entorno de ejecución local**.
    *   Pega la URL y conecta.
