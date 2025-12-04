@echo off
REM Ir a la carpeta del proyecto
cd /d "C:\Users\walte\Documents\ProyectoCom2"

REM Activar el entorno virtual
call venv\Scripts\activate.bat

REM Lanzar Streamlit
python -m streamlit run app_modulacion.py

REM Dejar la ventana abierta para ver errores si algo falla
pause


