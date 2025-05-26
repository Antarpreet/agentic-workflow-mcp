import requests

def api_fetch(
        url: str, method: str = "GET", headers: dict = None, params: dict = None, data: dict = None, json: dict = None,
        timeout: int = 10
    ) -> dict:
    """
    Fetch data from an API endpoint.

    Args:
        url (str): The API endpoint URL.
        method (str): HTTP method (GET, POST, etc.).
        headers (dict, optional): HTTP headers.
        params (dict, optional): URL query parameters.
        data (dict, optional): Form data for POST/PUT.
        json (dict, optional): JSON body for POST/PUT.
        timeout (int, optional): Request timeout in seconds.

    Returns:
        dict: JSON response from the API.

    Raises:
        requests.RequestException: If the request fails.
    """
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            data=data,
            json=json,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error: API fetch error: {e}")
        return None
