import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np

hide_streamlit_style = """
            <style>
            #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.markdown("<h1 style='text-align: center; color: red;'><a href=\"https://github.com/ronylpatil/Covid-19-Detection-WebApp-using-Deep-Hybrid-Learning/\">COVID DETECTION USING MRI\'s</a></h1>", unsafe_allow_html = True)
# st.markdown("""<a href="https://www.example.com/">example.com</a>""", unsafe_allow_html = True)

def main() :
    file_uploaded = st.file_uploader('Upload an MRI Image...', type = 'jpg')
    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded MRI.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result = predict_class(image)
        st.write('Prediction : {}'.format(result))
        if result == 'COVID-19' :
            st.write('Please consult a doctor as soon as possible.')
        else :
            st.write('There is no need to worry but if symptoms appear, consult a doctor as soon as possible.')

def predict_class(image) :
    with st.spinner('Loading, please be patient...') :
        # model = joblib.load(r'E:\DHL Project\CNN Projects\Deep Hybrid Learning Projects\X-ray\xray.pkl')
        model = joblib.load('xray.pkl')
        vggmodel = VGG16(weights = 'imagenet', include_top = False, input_shape = (256, 256, 3))
        for i in vggmodel.layers :
            i.trainable = False

    label = {0 : 'COVID-19', 1 : 'NORMAL'}

    # image = cv2.merge((image, image, image))
    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    # test_image = cv2.merge((test_image, test_image, test_image))
    test_image = np.concatenate((test_image,)*3, axis = -1)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    feature_extractor = vggmodel.predict(test_image)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    prediction = model.predict(features)[0]
    final = label[prediction]
    return final

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>

<div class="footer">
<p align="center"> <a href="https://www.linkedin.com/in/ronylpatil/">Developed with ❤ by ronil</a></p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)
if __name__ == '__main__' :
    main()
