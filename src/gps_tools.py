
import nvector as nv
import csv


class AnomalyLocation:
    def __init__(self, latitude, longitude, label):
        self.latitude = latitude
        self.longitude = longitude
        self.label = label


def convert_csv_gmaps(points: list[list[float]]):
    '''
        ## Convert locations  to CSV format for Google Maps
        Parameters
        ----------
        points : list of locations given in the [latitude, longitude] format  
    '''
    # csv header
    fieldnames = ['Name', 'Location', 'Description']
    rows = []
    # csv data
    for i, location in enumerate(points):
        rows.append({
            'Name': f"Point{i}",
            "Location": (location[0], location[1]),
            "Description": ""
        })
    # write to csv
    with open('points.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def interpolation(gps_location1: list[float], gps_location2: list[float], count: int) -> list[list[float]]:
    '''
        ## Interpolate between two GPS locations 
        Given two GPS locations, this function will return a list of GPS locations that are interpolated between the two locations.
        The number of locations to be obtained by interpolation is passed as the count parameter.

        Parameters
        ----------
        gps_location1 : GPS location1
        gps_location2 : GPS location2
        count: number of locations to be obtained by interpolation
    '''
    # create list of GPS locations
    new_points: list[list[float]] = []
    #  add location1 to list
    new_points.append(gps_location1)

    # create nvector frame
    wgs84 = nv.FrameE(name='WGS84')
    # convert location1 to nvector
    n_EB_E_t0 = wgs84.GeoPoint(
        gps_location1[0], gps_location1[1], degrees=True).to_nvector()
    # convert location2 to nvector
    n_EB_E_t1 = wgs84.GeoPoint(
        gps_location2[0], gps_location2[1], degrees=True).to_nvector()
    # path between location1 and location2
    path = nv.GeoPath(n_EB_E_t0, n_EB_E_t1)
    t0 = 10
    t1 = t0*(count+2)

    for i in range(2, count+2):
        ti = t0*i  # time of interpolation

        ti_n = (ti - t0) / (t1 - t0)  # normalized time of interpolation
        # interpolate between location1 and location2 at time ti
        g_EB_E_ti = path.interpolate(ti_n).to_geo_point()
        # convert nvector to  latitude, longotude GPS location
        lat_ti, lon_ti, _ = g_EB_E_ti.latlon_deg
        # add interpolated location to list
        new_points.append([lat_ti, lon_ti])
    # add location2 to list
    new_points.append(gps_location2)
    return new_points


a = interpolation([23.13102, -82.36181], [23.13369, -82.36078], 10)
convert_csv_gmaps(a)
print(a)
