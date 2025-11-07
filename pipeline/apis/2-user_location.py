#!/usr/bin/env python3
"""Return launch location from SpaceX API"""

import requests
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <URL>")
        sys.exit(1)

    res = requests.get(sys.argv[1])

    if res.status_code == 403:
        rate_limit = int(res.headers.get('X-Ratelimit-Reset', 0))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print(f"Reset in {diff} min")

    elif res.status_code == 404:
        print("Not found")

    elif res.status_code == 200:
        data = res.json()

        # SpaceX launches have a launchpad ID, we can use it to get the location
        if "launchpad" in data:
            launchpad_id = data["launchpad"]
            launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
            launchpad_data = requests.get(launchpad_url).json()
            print(launchpad_data["locality"])  # prints e.g., "Cape Canaveral"
        else:
            print("No launchpad info found")
