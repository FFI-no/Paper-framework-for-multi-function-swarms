import yaml, os, shelve, shutil
	
class Logger(object):
    def __init__(self, case, visualization=None, save_video=False, prefix_folder="logs"):
        self._viz = visualization
        self._save_video = save_video
        #Case name is assumed to be unique
        self._case_name = str(case)
        
        self._generate_filenames(prefix_folder)
        
        if os.path.exists(self._log_folder):
            shutil.rmtree(self._log_folder)
        os.makedirs(self._log_folder)
                
        self._dump_config(case.config)

        if self._save_video:
            self._video_folder = os.path.join(self._log_folder, "video")
            os.makedirs(self._video_folder)
        else:
            self._video_folder = None
            
        self._iteration = 0

        self._log_delay = 1.

    def set_log_delay(self, log_delay):
        self._log_delay = log_delay
        
    def _generate_filenames(self, folder):
        self._log_folder = os.path.join(folder, self._case_name)
        self._config_filename = os.path.join(self._log_folder, "config.txt")
        self._log_filename = os.path.join(self._log_folder, "simulation.log")
            
    def _dump_config(self, config):
        with open(self._config_filename, "w") as file_handle:
            yaml.dump(config, file_handle, default_flow_style=False)
            
    def _read_config(self):
        with open(self._config_filename, "r") as file_handle:
            return yaml.load(file_handle)
            
    def _read_logfile(self):
        with open(self._log_filename,"r") as logfile:
            data = logfile.read()
        return data
        
    def _dump_logfile(self, data):
        with open(self._log_filename,"w") as logfile:
            logfile.write(data)
        
    def update(self, current_time, case):
        if self._viz is not None:
            self._viz.update(self._iteration, case, folder=self._video_folder)
            
        self._simulation_log[str(self._iteration)] = case #sdata
        self._iteration += 1
        
        return [(current_time + self._log_delay, self.update)]
            
    def __enter__(self):
        self._simulation_log =  shelve.DbfilenameShelf(self._log_filename, protocol=2, writeback=False)
        
        return self
        
    def __exit__(self, type, value, traceback):        
        if getattr(self, '_viz', None) is not None :
            self._viz.close()
        self._simulation_log.close()

    def __getstate__(self):
        self._simulation_log.close()
        
        state = self.__dict__.copy()
        del state["_viz"]

        state["logfile_content"] =  self._read_logfile()
        state["configfile_content"] = self._read_config()
        
        shutil.rmtree(self._log_folder)
            
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        
        self._generate_filenames("logs")
        
        if os.path.exists(self._log_folder):
            shutil.rmtree(self._log_folder)
        os.makedirs(self._log_folder)
        
        self._dump_config(self.configfile_content)
        self._dump_logfile(self.logfile_content)
        
        del self.logfile_content
        del self.configfile_content