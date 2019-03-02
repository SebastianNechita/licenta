from pedect.input.inputProcessing import read_seq

class VideoHolder:

    def __init__(self, chosenDataset, setName, videoNr):
        video_path = chosenDataset.getVideoPath(setName, videoNr)
        self.video = read_seq(video_path)

    def getLength(self):
        return len(self.video)

    def getVideo(self):
        return self.video

    def getFrame(self, frameNr):
        return self.video[frameNr]