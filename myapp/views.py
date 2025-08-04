from django.shortcuts import render
from .ml_pipeline import predict_fare

def index(request):
    ridge_result = None
    dtree_result = None
    error = None

    if request.method == 'POST':
        try:
            # Get all 20 numeric input features
            car_condition = float(request.POST.get('car_condition'))
            weather = float(request.POST.get('weather'))
            traffic_condition = float(request.POST.get('traffic_condition'))
            pickup_longitude = float(request.POST.get('pickup_longitude'))
            pickup_latitude = float(request.POST.get('pickup_latitude'))
            dropoff_longitude = float(request.POST.get('dropoff_longitude'))
            dropoff_latitude = float(request.POST.get('dropoff_latitude'))
            passenger_count = float(request.POST.get('passenger_count'))
            hour = float(request.POST.get('hour'))
            day = float(request.POST.get('day'))
            month = float(request.POST.get('month'))
            weekday = float(request.POST.get('weekday'))
            year = float(request.POST.get('year'))
            jfk_dist = float(request.POST.get('jfk_dist'))
            ewr_dist = float(request.POST.get('ewr_dist'))
            lga_dist = float(request.POST.get('lga_dist'))
            sol_dist = float(request.POST.get('sol_dist'))
            nyc_dist = float(request.POST.get('nyc_dist'))
            distance = float(request.POST.get('distance'))
            bearing = float(request.POST.get('bearing'))

            # Input order must match model training
            input_features = [
                car_condition, weather, traffic_condition,
                pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
                passenger_count, hour, day, month, weekday, year,
                jfk_dist, ewr_dist, lga_dist, sol_dist, nyc_dist, distance, bearing
            ]

            # Get predictions
            ridge_pred, dtree_pred = predict_fare(input_features)

            ridge_result = round(ridge_pred, 2)
            dtree_result = round(dtree_pred, 2)

        except Exception as e:
            error = f"Error: {e}"

    return render(request, 'myapp/index.html', {
        'ridge_result': ridge_result,
        'dtree_result': dtree_result,
        'error': error
    })
