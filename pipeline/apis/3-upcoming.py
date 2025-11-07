#!/usr/bin/env python3
"""Pipeline Api"""
import requests
from datetime import datetime


if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)
    
    if r.status_code != 200:
        print("Error fetching launches data")
        exit(1)
    
    launches = r.json()
    
    if not launches:
        print("No upcoming launches found")
        exit(0)
    
    # Find the launch with the earliest date
    earliest_launch = min(launches, key=lambda x: x["date_unix"])
    
    launch_name = earliest_launch["name"]
    date = earliest_launch["date_local"]
    rocket_number = earliest_launch["rocket"]
    launchpad_number = earliest_launch["launchpad"]
    
    # Get rocket name
    rurl = f"https://api.spacexdata.com/v4/rockets/{rocket_number}"
    rocket_response = requests.get(rurl)
    if rocket_response.status_code == 200:
        rocket_name = rocket_response.json()["name"]
    else:
        rocket_name = "Unknown Rocket"
    
    # Get launchpad details
    lurl = f"https://api.spacexdata.com/v4/launchpads/{launchpad_number}"
    launchpad_response = requests.get(lurl)
    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data["name"]
        launchpad_local = launchpad_data["locality"]
    else:
        launchpad_name = "Unknown Launchpad"
        launchpad_local = "Unknown Location"
    
    # Format the output string
    output_string = f"{launch_name} ({date}) {rocket_name} - {launchpad_name} ({launchpad_local})"
    print(output_string)