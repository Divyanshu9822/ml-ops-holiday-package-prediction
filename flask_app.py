from flask import Flask, render_template, request
from config import Config
from src.logger import logger
from src.pipeline.prediction import PredictionPipeline
import os

app = Flask(__name__)
app.config.from_object(Config)


@app.route("/train", methods=["GET"])
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            typeofcontact = request.form["typeofcontact"]
            citytier = int(request.form["citytier"])
            durationofpitch = float(request.form["durationofpitch"])
            occupation = request.form["occupation"]
            gender = request.form["gender"]
            numberoffollowups = float(request.form["numberoffollowups"])
            productpitched = request.form["productpitched"]
            preferredpropertystar = float(request.form["preferredpropertystar"])
            maritalstatus = request.form["maritalstatus"]
            numberoftrips = float(request.form["numberoftrips"])
            passport = int(request.form["passport"])
            pitchsatisfactionscores = int(request.form["pitchsatisfactionscores"])
            owncar = int(request.form["owncar"])
            designation = request.form["designation"]
            monthlyincome = float(request.form["monthlyincome"])
            totalvisiting = float(request.form["totalvisiting"])

            model = PredictionPipeline()

            input_data = [
                [
                    age,
                    typeofcontact,
                    citytier,
                    durationofpitch,
                    occupation,
                    gender,
                    numberoffollowups,
                    productpitched,
                    preferredpropertystar,
                    maritalstatus,
                    numberoftrips,
                    passport,
                    pitchsatisfactionscores,
                    owncar,
                    designation,
                    monthlyincome,
                    totalvisiting,
                ]
            ]

            prediction = model.predict(input_data)

            logger.info("Received input values:")
            logger.info(f"Age: {age}")
            logger.info(f"Type of Contact: {typeofcontact}")
            logger.info(f"City Tier: {citytier}")
            logger.info(f"Duration of Pitch: {durationofpitch}")
            logger.info(f"Occupation: {occupation}")
            logger.info(f"Gender: {gender}")
            logger.info(f"Number of Follow-ups: {numberoffollowups}")
            logger.info(f"Product Pitched: {productpitched}")
            logger.info(f"Preferred Property Star: {preferredpropertystar}")
            logger.info(f"Marital Status: {maritalstatus}")
            logger.info(f"Number of Trips: {numberoftrips}")
            logger.info(f"Passport: {passport}")
            logger.info(f"Pitch Satisfaction Score: {pitchsatisfactionscores}")
            logger.info(f"Own Car: {owncar}")
            logger.info(f"Designation: {designation}")
            logger.info(f"Monthly Income: {monthlyincome}")
            logger.info(f"Total Visiting: {totalvisiting}")
            logger.info(f"Prediction: {prediction}")

            if prediction == 1:
                prediction_message = (
                    "The customer is likely to purchase the travel package."
                )
            else:
                prediction_message = (
                    "The customer is unlikely to purchase the travel package."
                )

            logger.info(f"Prediction: {prediction}")
            logger.info(f"Prediction Message: {prediction_message}")

            return render_template("index.html", prediction=prediction_message)

        except Exception as e:
            logger.info(e)
            return render_template(
                "index.html", prediction="Error: Unable to make prediction."
            )
    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
