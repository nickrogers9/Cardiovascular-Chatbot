@echo off
echo Starting Cardiovascular Chatbot Model Evaluation
echo ================================================

REM Check if vector store exists
if not exist ".\chroma_db_optimized" (
    echo ERROR: Vector store not found!
    echo Please run: python vector.py
    pause
    exit /b 1
)

REM Check if evaluation dataset exists
if not exist ".\evaluation_dataset.json" (
    echo ERROR: evaluation_dataset.json not found!
    echo Please create the evaluation dataset first.
    pause
    exit /b 1
)

REM Install requirements
echo Checking dependencies...
python install_requirements.py

REM Run evaluation
echo.
echo Starting model evaluation...
echo This may take 30-60 minutes depending on your system...
echo.
python evaluator.py

echo.
echo Evaluation complete! Check the generated files:
echo - evaluation_results_summary_*.json
echo - evaluation_detailed_*.csv
echo - model_evaluation_plots_*.png
echo - radar_chart_*.png
pause