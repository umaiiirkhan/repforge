"""Tests for the Flask API endpoints."""


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"GymBot" in res.data


def test_health(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}


def test_chat_returns_reply(client):
    res = client.post("/api/chat", json={"message": "hello"})
    data = res.get_json()
    assert res.status_code == 200
    assert data["intent"] == "greeting"
    assert {"reply", "intent", "confidence", "quick_replies", "source"} <= data.keys()


def test_chat_validation(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 400
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400
    assert client.post("/api/chat", json={"message": "x" * 501}).status_code == 400
    assert client.post("/api/chat", json={}).status_code == 400
    assert client.post("/api/chat", json={"message": 123}).status_code == 400
    assert client.post("/api/chat", data="not json", content_type="text/plain").status_code == 400


def test_chat_keeps_conversation_state_in_session(client):
    client.post("/api/chat", json={"message": "make me a workout plan"})
    client.post("/api/chat", json={"message": "beginner"})
    client.post("/api/chat", json={"message": "full gym"})
    data = client.post("/api/chat", json={"message": "4"}).get_json()
    assert "Gym Upper/Lower Plan" in data["reply"]


def test_reset_clears_state(client):
    client.post("/api/chat", json={"message": "make me a workout plan"})
    client.post("/api/reset")
    data = client.post("/api/chat", json={"message": "hello"}).get_json()
    assert data["intent"] == "greeting"


def test_repeated_requests(client):
    for _ in range(20):
        assert client.post("/api/chat", json={"message": "how do I warm up"}).status_code == 200


def test_logs_crud(client):
    assert client.get("/api/logs").get_json()["summary"]["sessions"] == 0
    res = client.post("/api/logs", json={"workout": "Full Body A", "duration_min": 45, "notes": "felt good"})
    assert res.status_code == 201
    log_id = res.get_json()["id"]
    data = client.get("/api/logs").get_json()
    assert data["summary"]["sessions"] == 1 and data["logs"][0]["workout"] == "Full Body A"
    assert client.delete(f"/api/logs/{log_id}").status_code == 200
    assert client.delete(f"/api/logs/{log_id}").status_code == 404


def test_logs_validation(client):
    assert client.post("/api/logs", json={"workout": "", "duration_min": 30}).status_code == 400
    assert client.post("/api/logs", json={"workout": "Run", "duration_min": 0}).status_code == 400
    assert client.post("/api/logs", json={"workout": "Run", "duration_min": "abc"}).status_code == 400
    assert client.post("/api/logs", json={"workout": "Run", "duration_min": 30, "date": "2999-01-01"}).status_code == 400
    assert client.post("/api/logs", json={"workout": "Run", "duration_min": 30, "date": "yesterday"}).status_code == 400


def test_users_cannot_see_each_others_logs(tmp_path):
    from app import create_app
    app = create_app({"TESTING": True, "DB_PATH": tmp_path / "shared.db", "SECRET_KEY": "t"})
    a, b = app.test_client(), app.test_client()
    a.get("/")
    b.get("/")
    a.post("/api/logs", json={"workout": "A's workout", "duration_min": 30})
    assert b.get("/api/logs").get_json()["logs"] == []


def test_delete_all_data(client):
    client.post("/api/logs", json={"workout": "Legs", "duration_min": 50})
    assert client.delete("/api/data").status_code == 200
    assert client.get("/api/logs").get_json()["logs"] == []


def test_unknown_api_route(client):
    assert client.get("/api/nope").status_code == 404
