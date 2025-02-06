import xarray as xr

U = xr.DataArray(
    [[1, 1], [1, 1]],
    name="U",
    dims=("i", "j"),
    coords={
        "lon": (("i", "j"), [[90, 270], [90, 270]]),
        "lat": (("i", "j"), [[-45, -45], [45, 45]]),
    },
)
V = xr.DataArray(
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    name="V",
    dims=("lon", "lat"),
    coords={"lon": [0, 90, 180, 270], "lat": [-45, 45]},
)


def interp_method_nD(da, lon=None, lat=None):
    # 2d grid logic here
    pass


def interp_method_1D_NN(da, lon=None, lat=None):
    return da.sel(lon=lon, lat=lat, method="nearest")


def interp_method_1D_lin(da, lon=None, lat=None):
    return da.interp(lon=lon, lat=lat)


def unit_conversion(): ...


class Field:
    def __init__(
        self,
        da=None,  #! grid??
        interp_method=None,
    ):
        self.da = da
        self.interp_method = interp_method

    def eval(self, *, lon=None, lat=None, target_unit="m/s"):
        return unit_conversion(
            self.interp_method(self.U.da, lon=lon, lat=lat),
            self.da.units,
            target_unit,
        )


class Fieldset:
    def __init__(self, fields: list[Field]):
        self.fields = fields

    def __getattr__(self, attr):
        return self.fields[attr]


fieldset = Fieldset(
    U=Field(U, interp_method_nD),
    V=Field(V, interp_method_1D_NN),
)

lon_poi = 30
lat_poi = 12

U_poi = fieldset.get_U(lon=lon_poi, lat=lat_poi)
V_poi = fieldset.get_V(lon=lon_poi, lat=lat_poi)
print(U_poi)
print(V_poi)
