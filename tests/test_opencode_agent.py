import os

from rgb_agent.agent.opencode_agent import _container_local_base_url, _resolve_analyzer_model


def test_container_local_base_url_uses_host_network_for_local_hostnames() -> None:
    base_url, network_mode = _container_local_base_url("http://host.docker.internal:1234/v1")
    assert base_url == "http://127.0.0.1:1234/v1"
    assert network_mode == "host"


def test_container_local_base_url_keeps_remote_urls_on_bridge() -> None:
    base_url, network_mode = _container_local_base_url("https://api.example.com/v1")
    assert base_url == "https://api.example.com/v1"
    assert network_mode == "bridge"


def test_resolve_local_analyzer_model_uses_container_safe_base_url(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_ANALYZER_BASE_URL", "http://host.docker.internal:1234/v1")
    monkeypatch.setenv("LOCAL_ANALYZER_MODEL_ID", "configured-model")
    monkeypatch.setattr(
        "rgb_agent.agent.opencode_agent._discover_local_model_id",
        lambda _: "Qwen/Qwen3.5-0.8B",
    )

    spec = _resolve_analyzer_model("local-qwen")

    assert spec.oc_model == "local/Qwen/Qwen3.5-0.8B"
    assert spec.provider_config["local"]["options"]["baseURL"] == "http://127.0.0.1:1234/v1"
    assert "Qwen/Qwen3.5-0.8B" in spec.provider_config["local"]["models"]
    assert spec.docker_network_mode == "host"
    assert spec.opencode_print_logs is True
    assert spec.opencode_log_level == "DEBUG"
    assert spec.harden_container is False
