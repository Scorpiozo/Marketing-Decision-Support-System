"""
Lightweight Flask API + static frontend for the Bank Marketing DSS.

    python run.py                 # http://127.0.0.1:5000

Endpoints (all GET unless noted; filter params may repeat, e.g. job=student&job=retired):

    /api/health
    /api/meta                                   filter options, models, form options
    /api/overview?<filters>                     historical KPIs + monthly conversion
    /api/insights/drivers?<filters>             conversion by month/contacts/outcome/duration
    /api/insights/segments?segment=job&<filters>
    /api/targeting/summary?model=&<filters>     lead KPIs, recommendation mix, histogram
    /api/targeting/customers?model=&<filters>&temperature=&fatigue=&q=&sort=&order=&page=&page_size=
    /api/targeting/export?...                   same filters as /customers, CSV download
    /api/customers/<id>?model=                  one test-set customer
    POST /api/predict?model=                    score one customer (JSON body)
    /api/models                                 metrics, ROC, confusion matrices, importances
"""

import os

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from werkzeug.exceptions import MethodNotAllowed, NotFound

from api.service import DashboardService, ValidationError
from src.config import PROJECT_ROOT

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


def _int_arg(name, default):
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValidationError(f"'{name}' must be an integer") from None


def create_app(service=None):
    """Build the app. The service (data + models) is created once per process."""
    app = Flask(__name__, static_folder=None)
    app.json.sort_keys = False
    app.config["SERVICE"] = service or DashboardService()

    def svc():
        return app.config["SERVICE"]

    def filters():
        return svc().parse_filters(request.args.getlist)

    def model():
        return svc().parse_model(request.args.get("model"))

    def targeting_args():
        return dict(
            temperature=request.args.get("temperature") or None,
            fatigue=request.args.get("fatigue") or None,
            query=request.args.get("q") or None,
            sort=request.args.get("sort", "probability"),
            order=request.args.get("order", "desc"),
        )

    @app.errorhandler(ValidationError)
    def bad_request(err):
        return jsonify(error=str(err)), 400

    @app.errorhandler(404)
    def not_found(_err):
        if request.path.startswith("/api/"):
            return jsonify(error=f"No API endpoint at {request.path}"), 404
        return send_from_directory(FRONTEND_DIR, "index.html"), 404

    @app.errorhandler(405)
    def method_not_allowed(_err):
        return jsonify(error=f"{request.method} is not allowed on {request.path}"), 405

    @app.after_request
    def cache_headers(resp):
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # API

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", startup_seconds=svc().startup_seconds,
                       production_model=svc().production_key)

    @app.get("/api/meta")
    def meta():
        return jsonify(svc().meta())

    @app.get("/api/overview")
    def overview():
        return jsonify(svc().overview(filters()))

    @app.get("/api/insights/drivers")
    def drivers():
        return jsonify(svc().drivers(filters()))

    @app.get("/api/insights/segments")
    def segments():
        return jsonify(svc().segment(filters(), request.args.get("segment", "job")))

    @app.get("/api/targeting/summary")
    def targeting_summary():
        return jsonify(svc().targeting_summary(model(), filters()))

    @app.get("/api/targeting/customers")
    def targeting_customers():
        return jsonify(svc().customers(
            model(), filters(), **targeting_args(),
            page=_int_arg("page", 1), page_size=_int_arg("page_size", 25),
        ))

    @app.get("/api/targeting/export")
    def targeting_export():
        key = model()
        csv_text, _n = svc().export_csv(key, filters(), **targeting_args())
        return Response(csv_text, mimetype="text/csv", headers={
            "Content-Disposition": f'attachment; filename="priority_customers_{key}.csv"'
        })

    @app.get("/api/customers/<int:record_id>")
    def customer(record_id):
        try:
            return jsonify(svc().customer(model(), record_id))
        except KeyError:
            return jsonify(error=f"No test-set customer with record ID {record_id}"), 404

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("Send the customer as a JSON object (Content-Type: application/json).")
        return jsonify(svc().predict(payload, model()))

    @app.get("/api/models")
    def models():
        return jsonify(svc().model_report())

    # Frontend (static files)

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path):
        if path.startswith("api/"):
            # An API path reached the static catch-all: 405 if the endpoint exists
            # for another method (e.g. GET /api/predict), else JSON 404.
            adapter = app.url_map.bind("localhost")
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                try:
                    adapter.match(f"/{path}", method=method)
                except (NotFound, MethodNotAllowed):
                    continue
                abort(405)
            abort(404)
        return send_from_directory(FRONTEND_DIR, path)

    return app
