from music21 import converter
s = converter.parse('abcnotation_midi/001-antafona_Bmaj.abc')
#s.show('midi')
s.write('midi',fp='001-antafona_Bmaj.mid') # deve ter formas de escolher instrumento e talz, mas vamos partir daqui