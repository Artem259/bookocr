import json


class PublicFieldsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, object):
            public_fields = {
                key: value
                for key, value in obj.__dict__.items()
                if not key.startswith("_") and not key.startswith("__")
            }
            return public_fields
        return super().default(obj)


class JSONPublicSerializable:
    @classmethod
    def __object_hook(cls, d):
        obj = cls()
        obj.__dict__.update(d)
        return obj

    def to_json(self, *args, **kwargs):
        return json.dumps(self, cls=PublicFieldsEncoder, *args, **kwargs)

    @classmethod
    def from_json(cls, json_str, *args, **kwargs):
        return json.loads(json_str, object_hook=cls.__object_hook, *args, **kwargs)

    def to_json_file(self, file_path, *args, **kwargs):
        json_data = self.to_json(*args, **kwargs)
        with open(file_path, "w") as f:
            f.write(json_data)

    @classmethod
    def from_json_file(cls, file_path, *args, **kwargs):
        with open(file_path, "r") as f:
            json_data = f.read()
        return cls.from_json(json_data, *args, **kwargs)
