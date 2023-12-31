import os, random, string

class Dataset:
    def __init__(self, folder="data/training_data"):
        # Setup storage folder
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
    
    def get_drives(self):
        driveIDList = {}
        for filename in os.listdir(self.folder):
            fileDriveID = filename.split('-')[1]
            if not(fileDriveID in driveIDList):
                driveIDList[fileDriveID] = []
            
            driveIDList[fileDriveID].append(filename)
        
        return driveIDList
    
    def get_drive_data(self, driveID):
        driveIDList = self.get_drives()
        driveIDFiles = driveIDList[driveID]
        
        driveData = [[None, None]] * len(driveIDFiles)
        for i in range(0, len(driveIDFiles)):
            filename = driveIDFiles[i]
            angle = int(filename.split('-')[0])
            frame = int(filename.split('-')[2].replace(".png", ""))
            fileData = [angle, self.folder + "/" + filename]
            driveData[frame] = fileData
        
        return driveData
    
    def get_frames(self, framecount=1000):
        files = os.listdir(self.folder)
        random.shuffle(files)
        if(len(files) > framecount):
            files = files[:framecount]
        return files