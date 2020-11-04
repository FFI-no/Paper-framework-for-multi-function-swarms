class Sensor(object):
    def __init__(self, agent, config):        
        self.position = agent.platform.position
        
    def update(self, platform, world):
        self.position = platform.position

    def get_patches(self):
        return []