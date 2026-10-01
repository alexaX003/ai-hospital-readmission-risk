def test_import_api():
    from src.api.main import app
    assert app.title == "Hospital Readmission Risk API"
