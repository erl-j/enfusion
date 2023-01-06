import soundfile as sf
import os

def export_sfz(outpath,regions,sample_rate):

    for region in regions:
        fn= str(region["midi_pitch_nr"])+".wav"
        fpath = os.path.join(outpath, "samples",fn)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        sf.write(fpath, region["waveform"], sample_rate)
    
    write_sfz_metadata(outpath+"/metadata.sfz",outpath,regions)


def write_sfz_metadata ( outpath, default_path, regions ): 
    """ 
    Exports the current instrument as an SFZ file. 
    """

    outstr="""
<control>
default_path=samples/
<global>
<group>\n""".format(default_path=default_path)

    for region in regions:
        outstr += "<region> sample={fn} lokey={lokey} hikey={hikey} pitch_keycenter={pitch_keycenter} \n".format(fn=str(region["midi_pitch_nr"])+".wav", lokey=region["lokey"], hikey=region["hikey"], pitch_keycenter=region["midi_pitch_nr"])

    with open(outpath, "w") as f:
        f.write(outstr)

    
    

