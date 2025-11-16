@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Starting Streamlit App...
streamlit run streamlit_app.py
pause

