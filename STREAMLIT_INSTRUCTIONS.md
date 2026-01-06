# How to Run the Explainability Consistency Analyzer App

1.  **Install Dependencies**
    Ensure you have all required packages installed. 
    *Note: Streamlit installation might take a few minutes.*
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**
    Execute the following command from the project root directory:
    ```bash
    streamlit run app.py
    ```

3.  **Using the App**
    *   **Upload**: Click "Browse files" to upload any CSV dataset.
    *   **Configure**: Select the target column (the variable you want to predict) and adjust model/noise settings in the sidebar.
    *   **Run**: Click "Run Analysis" to start the consistency evaluation.

## Notes
*   **Target Column**: Make sure to select the correct column you want to predict (e.g., 'Outcome', 'target', 'price').
*   **Preprocessing**: The app automatically handles non-numeric data using One-Hot Encoding and fills missing values with 0.
*   **Performance**: Large datasets may take longer to process; try using fewer seeds (e.g., 3) for a quick check.
