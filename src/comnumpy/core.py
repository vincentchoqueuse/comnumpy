import numpy as np
import copy
import json
from .analysers.recorder import Recorder

class Sequential():

    def __init__(self, module_list, name="sequential"):
        self.name = name
        self.module_list = module_list

    def get_module(self, index):
        return self.module_list[index]
    
    def get_module_by_name(self, module_name):
        """
        Retrieve a module from the module list by its name.

        Parameters
        ----------
        module_name : str
            The name of the module to retrieve.

        Returns
        -------
        object
            The module with the specified name, or None if not found.
        """
        for module in self.module_list:
            if hasattr(module, 'name'):
                if module.name == module_name:
                    return module
        return None  # Or raise an exception if you prefer

    def get_extra_dict(self):
        return {"name": self.name}

    def to_dict(self):
        module_dict = []
        for module in self.module_list:
            module_dict.append({"name": module.name, "parameters": module.to_dict()})

        extra_dict = self.get_extra_dict()
        return {"general" :extra_dict, "modules": module_dict}

    def to_json(self, path=None, indent=4):
        """Export data to json
        :path: String with filename. If None, the result is returned as a string.
        """
        if path:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=indent)
        else:
            return json.dumps(self.to_dict(), indent=indent)
        
    def add_recorders(self):
        """Add recorders between each processor """
        # Create a new list and insert the objects
        new_list = []
        for i, item in enumerate(self.module_list):
            new_list.append(item)

            if i < len(self.module_list) - 1:  # Avoid inserting after the last element
                name = "post_{}".format(item.name)
                recorder = Recorder(name=name)
                new_list.append(recorder)

        # save list
        self.module_list = new_list
        
    def forward(self, x):
        y = x
        for processor in self.module_list:
            y = processor(y)
        return y

    def __call__(self,x):
        return self.forward(x)
    
    def __getitem__(self, key):
        """
        Retrieve a module using the [] operator.

        Parameters
        ----------
        key : str
            The name of the module to retrieve.

        Returns
        -------
        object
            The module with the specified name, or None if not found.
        """
        return self.get_module_by_name(key)


class MIMO_Wrapper(Sequential):
    """A MIMO Wrapper encapsulates a siso equential list to apply for each input stream"""

    def __init__(self, module_list, N_t, name="sequential"):
        self.module_list = module_list
        self.N_t = N_t
        self.name = name

    def get_module(self, index):
        if isinstance(self.module_list, list):
            module = self.module_list[index]
        else:
            module = copy.deepcopy(self.module_list)
            if isinstance(module, Sequential):
                for submodule in module.module_list:
                    submodule.name = "{}_voice{}".format(submodule.name, index)
            else:
                module.name = "{}_voice{}".format(module.name, index)
    
        return module

    def forward(self,X):
        Y_list = []
        _, N = X.shape
        for index in range(self.N_t):
            sequential = self.get_module(index)
            x_temp = X[index,:] 
            y = sequential(x_temp)
            Y_list.append(y)

        Y = np.array(Y_list)
        return Y


