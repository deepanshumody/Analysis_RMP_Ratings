import streamlit as st

# Optional page config
st.set_page_config(
    page_title="My RMP Multi-Page App",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Welcome to the RateMyProfessors Analysis App")
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://media.giphy.com/media/KzJkzjggfGN5Py6nkT/giphy.gif" 
                 style="width: 200px; border-radius: 8px;" alt="Welcome GIF">
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        This interactive app explores **RateMyProfessors** data:
        - **Pepper Analysis** for answer questions about the data and hypothesis testing  
        - **Regression Analysis** to predict average ratings or difficulty  

        **To get started**, open the sidebar (left) and select one of the pages:
        1. *Pepper Analysis*  
        2. *Regression Analysis*  

        ---
        **How to Use**:
        1. Go to the left sidebar to navigate among pages.  
        2. On each page, follow the instructions for uploading data (or using preloaded data).  
        3. Explore the visualizations, run models, and examine the results.
        """
    )

    st.info(
        "Go to the 'Pages' in the left sidebar. "
        "There you'll see 'Pepper Analysis' and 'Regression Analysis'."
    )

if __name__ == "__main__":
    main()
