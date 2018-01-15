class sutdent(object):
    @property
    def score(self):
        return self._score
    @score.setter
    def score(self,value):
        if value<0 or value>100:
            raise ValueError("wrong range")
        self._score=value
s=sutdent()
s.score=39
print(s.score)
