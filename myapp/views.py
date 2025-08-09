from django.shortcuts import render
from .ml_pipeline import predict_fare

def index(request):
    xgb_result = None
    linear_result = None
    knn_result = None
    ridge_result = None
    dtree_result = None
    error = None

    if request.method == 'POST':
        try:
            # Collect exactly 9 numeric features
            input_features = [
                float(request.POST.get('pickup_longitude')),
                float(request.POST.get('pickup_latitude')),
                float(request.POST.get('dropoff_longitude')),
                float(request.POST.get('dropoff_latitude')),
                float(request.POST.get('passenger_count')),
                float(request.POST.get('hour')),
                float(request.POST.get('day')),
                float(request.POST.get('month')),
                float(request.POST.get('weekday')),
                float(request.POST.get('year')),
                float(request.POST.get('jfk_dist')),
                float(request.POST.get('ewr_dist')),
                float(request.POST.get('lga_dist')),
                float(request.POST.get('sol_dist')),
                float(request.POST.get('nyc_dist')),
                float(request.POST.get('distance')),
                float(request.POST.get('bearing ')),
            ]

            # Pass directly
            xgb_pred, linear_pred, ridge_pred, dtree_pred, knn_pred = predict_fare(input_features)


            # Round results
            xgb_result = round(xgb_pred, 2)
            linear_result = round(linear_pred, 2)
            knn_result = round(knn_pred, 2)
            ridge_result = round(ridge_pred, 2)
            dtree_result = round(dtree_pred, 2)

        except Exception as e:
            error = f"Error: {e}"

    return render(request, 'myapp/index.html', {
        'xgb_result': xgb_result,
        'linear_result': linear_result,
        'knn_result':knn_result,
        'ridge_result': ridge_result,
        'dtree_result': dtree_result,
        'error': error
    })
