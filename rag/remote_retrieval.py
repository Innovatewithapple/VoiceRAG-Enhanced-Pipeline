import requests
# We will replace this with the actual
# Colab/ngrok URL once the Colab server is running.
COLAB_URL = "https://plethora-registry-shrine.ngrok-free.dev"

def Retrieve_Remote(query, timeout=30):
    try:
        response = requests.post(
            f"{COLAB_URL}/query",
            json={
                "query": query
            },
            timeout=timeout,
        )

        response.raise_for_status()
        
        return response.json()

    except requests.exceptions.Timeout:
        print("⚠️ Remote retrieval timed out",flush=True)
        return None
    except requests.exceptions.ConnectionError:
        print("⚠️ Remote retrieval unavailable at the moment",flush=True)
        return None

    except requests.exceptions.HTTPError as e:
        print(f"⚠️ Remote retrieval HTTP error: {e}",flush=True)
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Remote retrieval failed: {e}",flush=True)
        return None