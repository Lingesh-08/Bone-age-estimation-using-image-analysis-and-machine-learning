import gradio as gr
import tempfile
from PIL import Image

from bone_age_pipeline import predict_bone_age


# ==========================================================
# MAIN PREDICTION FUNCTION
# ==========================================================

def run_prediction(
    image,
    gender,
    age_group
):

    # ------------------------------------------------------
    # SAVE TEMP IMAGE
    # ------------------------------------------------------

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    image.save(temp_file.name)

    # ------------------------------------------------------
    # GENDER BOOLEAN
    # ------------------------------------------------------

    gender_bool = True if gender == "Male" else False

    # ------------------------------------------------------
    # AGE GROUP MAPPING
    # ------------------------------------------------------

    age_group_map = {
        "Greater than 8 Years": "ge8",
        "Less than or Equal to 8 Years": "le8"
    }

    age_group = age_group_map[age_group]
    # ------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------

    result = predict_bone_age(
        image_path=temp_file.name,
        gender=gender_bool,
        age_group=age_group,
        dl_feature_dict={}
    )

    # ------------------------------------------------------
    # RETURN OUTPUT
    # ------------------------------------------------------

    return (
        f"{result['predicted_boneage_months']} months",
        f"{result['predicted_boneage_years']} years"
    )

custom_css = """
h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
}

.prose {
    font-size: 18px !important;
    line-height: 1.8 !important;
}
"""

# ==========================================================
# GRADIO UI
# ==========================================================

demo = gr.Interface(

    fn=run_prediction,

    inputs=[

        gr.Image(
            type="pil",
            label="Upload Hand X-Ray"
        ),

        gr.Radio(
            choices=["Male", "Female"],
            label="Gender"
        ),

        gr.Radio(
            choices=[
                "Greater than 8 Years",
                "Less than or Equal to 8 Years"
            ],
            label="Age Group"
        )
    ],

    outputs=[

        gr.Textbox(
            label="Predicted Bone Age (Months)"
        ),

        gr.Textbox(
            label="Predicted Bone Age (Years)"
        )
    ],
    

    title="Bone Age Prediction System",

    description="""
    Upload a pediatric hand X-ray image and select patient details:

    • Gender  
    • Age Group  

    Automated bone age estimation using anatomical and deep learning-based analysis.
    """,
    css=custom_css,
    allow_flagging="never"
)

# ==========================================================
# LAUNCH
# ==========================================================

if __name__ == "__main__":

    demo.launch()