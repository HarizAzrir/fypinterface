from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import mpld3
import streamlit.components.v1 as components
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import preprocessing
import readfile
# Create a title for the app
st.title("Sentiment Analysis App")

# Create a file uploader to allow the user to upload a slang dictionary
file2 = st.file_uploader("Upload a slang dictionary", type="csv")

# Create a file uploader to allow the user to upload a CSV file
file = st.file_uploader("Upload a CSV file", type="csv")

test = readfile.slangs()

# Check if a file has been uploaded
if file is not None:

    if file2 is not None:
        slang =  pd.read_csv(file2, index_col=0, header=None, squeeze=True).to_dict()
        test.set_slang(slang)
    # Read the CSV file into a DataFrame
    reviews = pd.read_csv(file)

    # Add a text input field for the user to specify the text column
    text_col = st.text_input("Text column name")

    # Add a checkbox for the user to choose whether to remove stop words
    remove_stopwords = st.checkbox("Remove stop words")

    # Show dataframe 
    st.dataframe(reviews["content"])

    # Add a button to trigger the preprocessing and evaluation process
    if st.button("Analyze"):
        # Perform the preprocessing and evaluation
        preprocessed_data = preprocessing.preprocess(reviews)
        st.success("Preprocessing complete!")
        st.dataframe(preprocessed_data["content_stem"].head(7))
        preprocessed_data = preprocessing.replaceslang(preprocessed_data)
        st.success("Replace Slang complete!")
        st.dataframe(preprocessed_data["content_stem"].head(7))
        preprocessed_data = preprocessing.replaceemoji(preprocessed_data)
        st.success("Replace emojis complete!")
        st.dataframe(preprocessed_data["content_stem"].head(10))
        preprocessed_data = preprocessing.replacestopwords(preprocessed_data)
        st.success(" Removing stopwords complete!")
        st.dataframe(preprocessed_data["content_stem"].head(7))
        preprocessed_data = preprocessing.replacestem_text(preprocessed_data)
        st.success("Stem text complete!")
        st.dataframe(preprocessed_data["content_stem"].head(7))
        
        train_data = preprocessing.train(preprocessed_data)
        test_data = preprocessing.test(preprocessed_data)
        st.success("Create Train Test Split Complete!")
        st.write(train_data.shape, test_data.shape)
        #evaluation_results = evaluate(preprocessed_data)

        # Display the results

        st.write("Evaluation results:")
        #st.write(evaluation_results)

         # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### First Chart")
            image = Image.open("meaningfulwords.png")
            st.image(image, caption='Number of meaningful words')
            
        with fig_col2:
            st.markdown("### Second Chart")
            image2 = Image.open("averagenostars.png")
            st.image(image2, caption='Average Number of stars')

        fig_col3, fig_col4 = st.columns(2)
        with fig_col3:
            st.markdown("### Third Chart")
            image3 = Image.open("negwordcloud.png")
            st.image(image3, caption='Negative Wordcloud')

        with fig_col4:
            st.markdown("### Fourth Chart")
            image4 = Image.open("poswordcloud.png")
            st.image(image4, caption='Positive Wordcloud')
 
        st.markdown("### Fifth Chart")
        image5 = Image.open("unigram.png")
        st.image(image5, caption='Top 20 Uni-grams in Negative Reviews')
 
        st.markdown("### Sixth Chart")
        image6 = Image.open("unigram2.png")
        st.image(image6, caption='Top 20 Uni-grams in Negative Reviews')


        df = preprocessing.vader(train_data)
        st.success("Sentiment Analysis complete!")
        st.dataframe(df.head(10))

        # Create empty lists to append the accuracy score for each compound score threshold level
        threshold = []
        accuracy = []

        # Loop through a range of compound scores from -1 to +1 and calculate the accuracy score for each compound score
        for i in np.linspace(-1, 1, 1000):
            vader_prediction = df['compound'].map(lambda x: 0 if x > i else 1)
            score = accuracy_score(df['target'], vader_prediction)
            threshold.append(i)
            accuracy.append(score)

        fig6 = plt.figure()
        plt.plot(threshold, accuracy, linewidth=3)
        plt.plot([-1,threshold[np.argmax(accuracy)]],[max(accuracy),max(accuracy)], linestyle='dashed', color='#F5B041')
        plt.plot([threshold[np.argmax(accuracy)],threshold[np.argmax(accuracy)]],[0.3,max(accuracy)], linestyle='dashed', color='#F5B041')
        plt.title("Accuracy Score by VADER's Compound Score Threshold", size=15, weight='bold')
        plt.ylabel('Accuracy score', size=12)
        plt.xlabel('Compound score threshold', size=12)
        st.pyplot(fig6)

        st.write("VADER's best accuracy score: ", max(accuracy))
        st.write("Compound score threshold ", threshold[np.argmax(accuracy)])

        fig7 = plt.figure()
        # Plot a histogram of the compound scores for negative reviews
        df[df['target']==1]['compound'].hist(grid=False, color='red', alpha=0.5, bins=30)

        # Plot a histogram of the compound scores for positive reviews
        df[df['target']==0]['compound'].hist(grid=False, color='green', alpha=0.5, bins=30)

        # Plot a vertical line to show the corresponding compound score threshold for the best accuracy score
        plt.plot([threshold[np.argmax(accuracy)],threshold[np.argmax(accuracy)]],[0,500], linestyle='dashed', linewidth=3, color='#F5B041')

        plt.title("VADER's Compound Score for Positive and Negative Reviews", size=15, weight='bold')
        plt.legend(['Threshold for best accuracy','Negative reviews', 'Positive reviews'], fontsize=12)
        plt.ylabel('Frequency', size=12)
        plt.xlabel('Compound scores', size=12)
        st.pyplot(fig7)

        # Prediction with compound threshold of 0.175
        df['vader_prediction'] = df['compound'].map(lambda x: 0 if x > threshold[np.argmax(accuracy)] else 1)
        # Confusion matrix
        cm = confusion_matrix(df['target'], df['vader_prediction'])
        cm_df = pd.DataFrame(cm, columns=['Predicted Positive Review','Predicted Negative Review'], index=['Actual Positive Review', 'Actual Negative Review'])
        st.dataframe(cm_df)
        