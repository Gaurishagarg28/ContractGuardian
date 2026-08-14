def confidence_level(confidence):

    if confidence >= 0.80:

        return "HIGH"

    elif confidence >= 0.60:

        return "MEDIUM"

    elif confidence >= 0.40:

        return "LOW"

    else:

        return "VERY LOW"