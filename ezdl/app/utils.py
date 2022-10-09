from ruamel.yaml import YAML


class ManipulationInputs:
    def __init__(self):
        self.updated_meta = None
        self.path = None
        self.filters = {}
        self.updated_config = None
        self.keys_to_delete = None
        self.files_to_delete = None
        self.artifacts_to_delete = None
        self.fix_string_params = None

    def update(self, path, filters, updated_config, updated_meta, to_delete, fix_string_params):
        if path:
            self.path = path

        if filters:
            self.filters = YAML(typ="safe", pure=True).load(filters)
        if updated_config:
            self.updated_config = YAML(typ="safe", pure=True).load(updated_config)
        if updated_meta:
            self.updated_meta = YAML(typ="safe", pure=True).load(updated_meta)

        if to_delete:
            dict_to_delete = YAML(typ="safe", pure=True).load(to_delete)
            self.keys_to_delete = dict_to_delete['keys']
            self.files_to_delete = dict_to_delete['files']
            self.artifacts_to_delete = dict_to_delete['artifacts']

        self.fix_string_params = fix_string_params
