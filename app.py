import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Set page configuration
st.set_page_config(
    page_title="Indian Monuments Classifier",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling with pastel colors
st.markdown(
    """
    <style>
    body {
        background-color: #FAF9F6;
        font-family: "Arial", sans-serif;
    }
    .main {
        background-color: #FFFFFF;
        color: #333333;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #89CFF0;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 20px;
        color: #B39EB5;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-section {
        background-color: #FFF8DC;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #999999;
        margin-top: 30px;
    }
    .info-box {
        background-color: #FFEFDB;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Header
st.markdown("<div class='title'>Indian Monuments Classifier 🕌</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Explore the beauty and history of India's architectural marvels!</div>", unsafe_allow_html=True)

# Sidebar for Info
with st.sidebar:
    st.title("Navigation")
    st.markdown("### About the App")
    st.write("This app classifies images of Indian monuments and provides detailed historical and architectural information.")
    st.markdown("### How to Use")
    st.write("1. Upload an image of a monument.\n2. Wait for the model to classify it.\n3. Explore the monument's detailed explanation.")

# Image Upload Section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("### Upload an Image of an Indian Monument")

uploaded_file = st.file_uploader("Choose an image file (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
   

    # Load the trained model (replace with your model's path)
    model = tf.keras.models.load_model('monument_classifier_model_densenet.h5')

    # Preprocess the image to match input requirements for the model
    st.markdown("### Prediction")
    with st.spinner("Classifying the image... Please wait!"):
        
        # Resize and preprocess the image for the model
        img = image.resize((128, 128))  # Adjust this size according to your model's input size
        img_array = np.array(img) / 255.0  # Normalize the image (if required by the model)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)

        # Decode the prediction (depends on your model output)
        # Assuming model outputs class probabilities and you have a predefined list of classes
        class_names =  ['Ajanta Caves', 'Akshardham Temple', 'Alai Darwaza', 'Amer Fort','Chhatrapati Shivaji Terminus', 'City Palace', 'Gateway of India',
                        'Gol Gumbaz', 'Golden Temple', 'Hawa Mahal', 'Jaisalmer Fort','Lotus Temple', 'Mahabodhi Temple', 'Meenakshi Temple', 
                        'Nalanda University Ruins', 'Qutub Minar', 'Sanchi Stupa','Sun Temple', 'Taj Mahal', 'Victoria Memorial']  # Add your classes here
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"🎉 The uploaded image is classified as: **{predicted_class}**")

        # Detailed Architecture Explanation
        monument_info = {
    "Ajanta Caves": {
        "description": "The Ajanta Caves are rock-cut Buddhist cave monuments in Maharashtra, known for their exquisite murals, paintings, and sculptures depicting the life of Buddha.",
        "architectural_features": [
            "30 rock-cut caves built between the 2nd century BCE and 480 CE.",
            "Depictions of Jataka tales and intricate carvings.",
            "Stupas and chaityas showcasing Buddhist architectural style."
        ],
        "fun_fact": "Ajanta Caves were rediscovered in 1819 by a British officer.",
        "architecture": "Buddhist Architecture"
    },
    "Hawa Mahal": {
        "description": "Hawa Mahal, or 'Palace of Winds,' in Jaipur is an iconic pink sandstone structure designed for royal ladies to observe street festivals.",
        "architectural_features": [
            "Five-story building with 953 small windows (jharokhas).",
            "Built with red and pink sandstone.",
            "Intricate latticework for ventilation and privacy."
        ],
        "fun_fact": "Its unique design keeps the interiors cool even during summers.",
        "architecture": "Rajput Architecture"
    },
    "Sanchi Stupa": {
        "description": "The Sanchi Stupa in Madhya Pradesh is one of the oldest stone structures in India and a significant Buddhist monument.",
        "architectural_features": [
            "Built around 3rd century BCE.",
            "Dome-shaped structure (Stupa) symbolizing Buddha's parinirvana.",
            "Carvings depicting Buddhist themes and the life of Buddha."
        ],
        "fun_fact": "It is a UNESCO World Heritage Site and an important pilgrimage destination for Buddhists.",
        "architecture": "Buddhist Architecture"
    },
    "Mahabodhi Temple": {
        "description": "Located in Bodh Gaya, Bihar, Mahabodhi Temple is one of the most important Buddhist pilgrimage sites, marking the location where Buddha attained enlightenment.",
        "architectural_features": [
            "A tall, pyramidal spire (Shikhara) above the sanctum.",
            "Carvings and sculptures depicting scenes from Buddha's life.",
            "A sacred Bodhi tree under which Buddha meditated."
        ],
        "fun_fact": "The temple has been a site of pilgrimage for over 2,000 years.",
        "architecture": "Buddhist Architecture"
    },
    "Qutub Minar": {
        "description": "Qutub Minar is a towering minaret located in Delhi, constructed during the Delhi Sultanate period, and is a UNESCO World Heritage Site.",
        "architectural_features": [
            "73-meter tall tower with intricate carvings.",
            "Made of red sandstone with inscriptions in Arabic.",
            "Built by Qutb-ud-Din Aibak in the 12th century."
        ],
        "fun_fact": "It is the tallest brick minaret in the world.",
        "architecture": "Indo-Islamic Architecture"
    },
    "Alai Darwaza": {
        "description": "Alai Darwaza is a gateway located within the Qutub complex in Delhi, known for its Indo-Islamic architectural style.",
        "architectural_features": [
            "Constructed in 1311 by Sultan Ala-ud-Din Khilji.",
            "Built using red sandstone and decorated with Islamic calligraphy.",
            "Exquisite arches and domed structure."
        ],
        "fun_fact": "It was the first example of true Islamic architecture in India.",
        "architecture": "Indo-Islamic Architecture"
    },
    "Gol Gumbaz": {
        "description": "Gol Gumbaz is a mausoleum located in Bijapur, Karnataka, known for its massive dome and unique acoustics.",
        "architectural_features": [
            "The dome, measuring 44 meters in diameter, is one of the largest in the world.",
            "The building features a circular gallery at the top, allowing sound to echo.",
            "Built by Sultan Muhammad Adil Shah in the 17th century."
        ],
        "fun_fact": "The dome's acoustics are so precise that a whisper can be heard clearly at the other end of the gallery.",
        "architecture": "Indo-Islamic Architecture"
    },
    "Gateway of India": {
        "description": "The Gateway of India is an iconic monument in Mumbai, built to commemorate the visit of King George V to India in 1911.",
        "architectural_features": [
            "Built in Indo-Saracenic architectural style.",
            "Made of yellow basalt and reinforced concrete.",
            "Designed by George Wittet, combining Hindu and Islamic elements."
        ],
        "fun_fact": "It was the site of the last British troops leaving India in 1948, marking the end of British rule.",
        "architecture": "Colonial British Architecture"
    },
    "Amer Fort": {
        "description": "Amer Fort, located in Jaipur, is a majestic hilltop fort known for its blend of Rajput and Mughal architectural styles.",
        "architectural_features": [
            "Features a series of gates, palaces, and courtyards.",
            "Intricate frescoes and mirror work.",
            "Strategically located on a hill with views of the surrounding landscape."
        ],
        "fun_fact": "It was originally built by Raja Man Singh in the 16th century and expanded by his successors.",
        "architecture": "Rajput Architecture"
    },
    "City Palace": {
        "description": "City Palace in Jaipur is a grand palace complex that showcases a fusion of Rajput and Mughal architectural styles.",
        "architectural_features": [
            "Intricate latticework, courtyards, and gardens.",
            "Walls adorned with colorful frescoes.",
            "Blends traditional Rajput and Mughal architectural elements."
        ],
        "fun_fact": "Parts of the palace are still occupied by the royal family of Jaipur.",
        "architecture": "Rajput Architecture"
    },
    "Jaisalmer Fort": {
        "description": "Jaisalmer Fort, also known as Sonar Quila, is a massive fort built in the 12th century in Rajasthan, known for its golden-yellow sandstone.",
        "architectural_features": [
            "A living fort with temples, palaces, and houses inside.",
            "Built using yellow sandstone, which gives it a golden hue.",
            "Intricate carvings and Jain temples."
        ],
        "fun_fact": "It is one of the largest fully preserved fortified cities in the world.",
        "architecture": "Rajput Architecture"
    },
    "Meenakshi Temple": {
        "description": "The Meenakshi Temple in Madurai is a stunning example of Dravidian architecture, dedicated to the goddess Meenakshi.",
        "architectural_features": [
            "A towering gopuram (gateway) adorned with colorful sculptures.",
            "Vast temple complex with shrines and halls.",
            "Intricate carvings depicting Hindu mythology."
        ],
        "fun_fact": "The temple is dedicated to Meenakshi, the goddess of fish, and her consort Sundareshwarar, an incarnation of Lord Shiva.",
        "architecture": "Dravidian Architecture"
    },
    "Sun Temple": {
        "description": "The Sun Temple at Konark in Odisha is a UNESCO World Heritage Site, known for its chariot-shaped structure dedicated to the Sun God.",
        "architectural_features": [
            "Built in the form of a colossal chariot with 24 wheels.",
            "Intricate stone carvings depicting scenes from Hindu mythology.",
            "Dedicated to Surya, the Sun God."
        ],
        "fun_fact": "The temple was once known as the Black Pagoda by European sailors due to its dark color.",
        "architecture": "Kalinga Architecture"
    },
    "Victoria Memorial": {
        "description": "Victoria Memorial in Kolkata is a grand white-marble monument built in honor of Queen Victoria, blending British and Mughal architectural styles.",
        "architectural_features": [
            "A massive marble dome with colonnades.",
            "British and Mughal architectural styles.",
            "Gardens and sculptures surrounding the memorial."
        ],
        "fun_fact": "It houses a museum and a collection of historical artifacts from the British colonial period.",
        "architecture": "Indo-Saracenic Architecture"
    },

    "Akshardham Temple": {
        "description": "The Akshardham Temple in Delhi is a sprawling Hindu temple complex that showcases India's ancient architecture, traditions, and spirituality.",
        "architectural_features": [
            "Constructed using pink sandstone and white marble.",
            "Carvings of deities, flora, and fauna.",
            "The main monument is intricately sculpted and features traditional Hindu temple architecture."
        ],
        "fun_fact": "It was built without the use of steel and features the largest stepwell-style water body in the world.",
        "architecture": "Hindu Temple Architecture"
    },
    "Chhatrapati Shivaji Terminus": {
        "description": "Chhatrapati Shivaji Terminus, formerly known as Victoria Terminus, is a historic railway station in Mumbai, combining Gothic and Indian architectural elements.",
        "architectural_features": [
            "Constructed with a blend of Victorian Gothic Revival and traditional Indian architecture.",
            "Features turrets, pointed arches, and detailed stone carvings.",
            "Designed by Frederick William Stevens and completed in 1887."
        ],
        "fun_fact": "It is one of the busiest railway stations in India and a UNESCO World Heritage Site.",
        "architecture": "Colonial British Architecture"
    },
    "Lotus Temple": {
        "description": "The Lotus Temple in Delhi is a modern architectural marvel shaped like a blooming lotus flower, serving as a Bahá'í House of Worship.",
        "architectural_features": [
            "Constructed using white marble, with 27 petal-like structures forming the lotus shape.",
            "Surrounded by nine pools of water that create a floating effect.",
            "Known for its simplicity and emphasis on prayer and meditation."
        ],
        "fun_fact": "The temple has received numerous awards for its architectural excellence and welcomes people from all religions.",
        "architecture": "Modern Architecture"
    },
    "Nalanda University Ruins": {
        "description": "The ruins of Nalanda University in Bihar represent the remains of one of the world's oldest universities, a center of learning in ancient India.",
        "architectural_features": [
            "Features stupas, monasteries, and classrooms constructed with red bricks.",
            "Intricate carvings and sculptures showcasing Buddhist art.",
            "Spread over a large area with a structured layout for residential and educational purposes."
        ],
        "fun_fact": "Nalanda University was an important seat of learning for over 800 years, attracting scholars from across the world.",
        "architecture": "Buddhist Architecture"
    },
    "Taj Mahal": {
        "description": "The Taj Mahal, located in Agra, is a world-famous mausoleum built by Emperor Shah Jahan in memory of his wife Mumtaz Mahal.",
        "architectural_features": [
            "Constructed with white marble, featuring intricate inlay work with precious stones.",
            "A large central dome flanked by four minarets.",
            "Reflecting pools and gardens following Charbagh layout."
        ],
        "fun_fact": "It is one of the New Seven Wonders of the World and a UNESCO World Heritage Site.",
        "architecture": "Indo-Islamic Architecture"
    },
    "Golden Temple": {
    "description": "The Golden Temple, also known as Harmandir Sahib, is a revered Sikh gurdwara located in Amritsar, Punjab, and serves as the spiritual center of Sikhism.",
    "architectural_features": [
        "Constructed with a combination of white marble and gold-plated panels.",
        "A central sanctum surrounded by a sacred pool (Amrit Sarovar).",
        "Features a blend of Sikh, Hindu, and Islamic architectural styles."
    ],
    "fun_fact": "The Golden Temple's dome is gilded with approximately 750 kg of pure gold, making it a dazzling spectacle.",
    "architecture": "Sikh Architecture"
}
}


        if predicted_class in monument_info:
            info = monument_info[predicted_class]
            st.markdown(f"**About the Monument:** {info['description']}")
            st.markdown("#### Architectural Features:")
            for feature in info["architectural_features"]:
                st.markdown(f"- {feature}")
            st.markdown(f"**Fun Fact:** {info['fun_fact']}")

            # Architecture section with highlighted styling
            st.markdown("<div class='architecture-section'>", unsafe_allow_html=True)
            st.markdown(f"### Architecture: {info['architecture']}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Sorry, we don't have detailed information for this monument yet.")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Developed with ❤️</div>", unsafe_allow_html=True)
