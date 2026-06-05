# 'import' tells Python to bring in an external toolkit (library) into our script.
# We are bringing in the 'streamlit' library and giving it a shorter nickname 'st' 
# so we don't have to type out the whole word every single time we use it.
import streamlit as st

# 'st.title' is a built-in Streamlit tool that creates a giant, bold header text 
# at the very top of your web page.
st.title("Project: Beat the House")

# 'st.subheader' creates a slightly smaller header directly underneath your main title.
# It's perfect for a subtitle or a project description.
st.subheader("MLB Sports Betting Model")

# 'st.write' is the all-purpose swiss-army knife command in Streamlit.
# It takes whatever text you type inside the quotes and prints it out as standard body paragraph text
st.write("Welcome! This is the skeleton layout for my 597 project.")

# Here, we are creating a split layout using tabs on the website.
# 'st.tabs' takes a list of names inside the square brackets [] and builds a tab menu.
# We assign those two tabs to two custom variables we made up: 'tab1' and 'tab2'.
tab1, tab2 = st.tabs(["Today's Matchup", "Bankroll Tracker"])

# 'with tab1:' tells Python: "Anything that is indented directly underneath this line 
# should only show up when the user clicks on the first tab ('Today's Matchups')."
with tab1:
    st.write("Placeholder: Daily game slates and model odds will display here.")

# 'with tab2:' tells Python: "Anything indented directly underneath this line
# should only show up when the user clicks on the second tab ('Bankroll Tracker')."
with tab2:
    st.write("Placeholder: Poisson vs. Baseline Random model bankrolls will display here.")


# Command to run the website: streamlit run app.py
# Website URL: http://localhost:8501