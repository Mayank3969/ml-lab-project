"""
mappings.py — UK STATS19 Categorical Code Mappings
Human-readable labels mapped to numeric codes used by models.
"""

CASUALTY_CLASS_MAP = {
    "Driver or Rider": 1,
    "Passenger": 2,
    "Pedestrian": 3
}

CASUALTY_TYPE_MAP = {
    "Pedestrian": 0,
    "Cyclist": 1,
    "Motorcycle (<=50cc)": 2,
    "Motorcycle (<=125cc)": 3,
    "Motorcycle (125cc-500cc)": 4,
    "Motorcycle (>500cc)": 5,
    "Taxi / Private hire car": 8,
    "Car": 9,
    "Minibus (8-16 passenger seats)": 10,
    "Bus or coach (17+ passenger seats)": 11,
    "Van / Goods vehicle (<=3.5 tonnes)": 19,
    "Goods vehicle (3.5t - 7.5t)": 20,
    "Heavy Goods Vehicle (>=7.5 tonnes)": 21,
    "Mobility scooter": 22,
    "Other vehicle": 90,
    "Unknown / Missing": -1
}

DAY_OF_WEEK_MAP = {
    "Sunday": 1,
    "Monday": 2,
    "Tuesday": 3,
    "Wednesday": 4,
    "Thursday": 5,
    "Friday": 6,
    "Saturday": 7
}

ROAD_CLASS_MAP = {
    "Motorway": 1,
    "A(M)": 2,
    "A": 3,
    "B": 4,
    "C": 5,
    "Unclassified": 6,
    "Missing / Unknown": -1
}

ROAD_TYPE_MAP = {
    "Roundabout": 1,
    "One way street": 2,
    "Dual carriageway": 3,
    "Single carriageway": 6,
    "Slip road": 7,
    "Unknown": 9
}

SPEED_LIMIT_OPTIONS = [20, 30, 40, 50, 60, 70]

JUNCTION_DETAIL_MAP = {
    "Not at or within 20 metres of junction": 0,
    "T or staggered junction": 13,
    "Crossroads": 16,
    "Multiple junction": 17,
    "Using private drive or entrance": 18,
    "Other junction": 19,
    "Unknown / Missing": -1
}

JUNCTION_CONTROL_MAP = {
    "Authorised person": 1,
    "Auto traffic signal": 2,
    "Stop sign": 3,
    "Give way or uncontrolled": 4,
    "Not at a junction / Unknown": 9,
    "Missing": -1
}

LIGHT_CONDITIONS_MAP = {
    "Daylight": 1,
    "Darkness - lights lit": 4,
    "Darkness - lights unlit": 5,
    "Darkness - no lighting": 6,
    "Darkness - lighting unknown": 7,
    "Missing": -1
}

WEATHER_CONDITIONS_MAP = {
    "Fine no high winds": 1,
    "Raining no high winds": 2,
    "Snowing no high winds": 3,
    "Fine + high winds": 4,
    "Raining + high winds": 5,
    "Snowing + high winds": 6,
    "Fog or mist": 7,
    "Other": 8,
    "Unknown": 9
}

ROAD_SURFACE_CONDITIONS_MAP = {
    "Dry": 1,
    "Wet or damp": 2,
    "Snow": 3,
    "Fine ice": 4,
    "Flood over 3cm deep": 5,
    "Missing / Unknown": 9
}

SPECIAL_CONDITIONS_MAP = {
    "None": 0,
    "Auto traffic signal out": 1,
    "Auto signal partially defective": 2,
    "Permanent road signing defective": 3,
    "Roadworks": 4,
    "Road surface defective": 5,
    "Oil or diesel": 6,
    "Mud": 7,
    "Missing / Unknown": -1
}

CARRIAGEWAY_HAZARDS_MAP = {
    "None": 0,
    "Dislodged vehicle load": 11,
    "Other object in carriageway": 12,
    "Involvement with previous accident": 13,
    "Dog on road": 14,
    "Other animal on road": 15,
    "Pedestrian in carriageway (not injured)": 16,
    "Missing / Unknown": -1
}

URBAN_RURAL_MAP = {
    "Urban": 1,
    "Rural": 2,
    "Unallocated": 3
}

SEVERITY_LABELS = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

SEVERITY_COLORS = {
    1: "#dc2626", # Red
    2: "#ea580c", # Amber/Orange
    3: "#16a34a"  # Green
}

SEVERITY_ICONS = {
    1: "🚨",
    2: "⚠️",
    3: "🟢"
}
