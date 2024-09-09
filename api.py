from flask import Flask, request, jsonify
from config import Config
from src.logger import logger
from src.pipeline.prediction import PredictionPipeline
import os

app = Flask(__name__)
app.config.from_object(Config)


@app.route("/train", methods=["GET"])
def train():
    try:
        os.system("python main.py")
        return "Training Successful!"
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return "Training failed!", 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "age",
            "typeofcontact",
            "citytier",
            "durationofpitch",
            "occupation",
            "gender",
            "numberoffollowups",
            "productpitched",
            "preferredpropertystar",
            "maritalstatus",
            "numberoftrips",
            "passport",
            "pitchsatisfactionscores",
            "owncar",
            "designation",
            "monthlyincome",
            "totalvisiting",
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        age = float(data["age"])
        typeofcontact = data["typeofcontact"]
        citytier = int(data["citytier"])
        durationofpitch = float(data["durationofpitch"])
        occupation = data["occupation"]
        gender = data["gender"]
        numberoffollowups = float(data["numberoffollowups"])
        productpitched = data["productpitched"]
        preferredpropertystar = float(data["preferredpropertystar"])
        maritalstatus = data["maritalstatus"]
        numberoftrips = float(data["numberoftrips"])
        passport = int(data["passport"])
        pitchsatisfactionscores = int(data["pitchsatisfactionscores"])
        owncar = int(data["owncar"])
        designation = data["designation"]
        monthlyincome = float(data["monthlyincome"])
        totalvisiting = float(data["totalvisiting"])

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

        pipeline = PredictionPipeline()

        prediction = pipeline.predict(input_data)

        print(prediction)

        prediction_message = (
            "The customer is likely to purchase the travel package."
            if prediction == 1
            else "The customer is unlikely to purchase the travel package."
        )

        print(prediction_message)

        response = {"prediction": int(prediction), "prediction_message": prediction_message}

        return jsonify(response)

    except Exception as e:
        logger.error(e)
        return jsonify({"error": "Unable to make prediction."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
