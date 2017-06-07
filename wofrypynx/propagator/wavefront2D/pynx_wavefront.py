

import numpy
import scipy.constants as codata

m_to_eV = codata.h*codata.c/codata.e
#angstroms_to_eV = codata.h*codata.c/codata.e*1e10


from pynx.wavefront.wavefront import Wavefront
from pynx.wavefront.operator import ImshowRGBA
from pynx.wavefront.cl_operator import FromPU, FreeFromPU, ToPU, PropagateNearField, PropagateFarField, PropagateFRT, default_processing_unit_wavefront


from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.decorators import WavefrontDecorator


class PYNXWavefront(Wavefront, WavefrontDecorator):

    def __init__(self,d=None, z=0, pixel_size=55e-6, wavelength=12398.4e-10 / 8000, copy_d=True):
        Wavefront.__init__(self,d=d, z=z, pixel_size=pixel_size, wavelength=wavelength, copy_d=copy_d)


    def get_wavelength(self):
        return self.wavelength

    def toGenericWavefront(self):
        x,y = self.get_x_y()
        wavefront = GenericWavefront2D.initialize_wavefront_from_range(x.min(),
                                                                       x.max(),
                                                                       y.min(),
                                                                       y.max(),
                                                                       number_of_points=(x.shape[0], y.shape[0]),
                                                                       wavelength=self.wavelength)

        print("Shape", wavefront.size())
        print("WL", m_to_eV, wavefront.get_wavelength(), wavefront.get_photon_energy())

        wavefront.set_complex_amplitude((numpy.fft.fftshift(self.d)).T)

        return wavefront

    @classmethod
    def fromGenericWavefront(cls, wavefront):
        x = wavefront.get_coordinate_x()
        ca = wavefront.get_complex_amplitude()
        return Wavefront(d=numpy.fft.ifftshift(ca.T), z=0, pixel_size=(x[1]-x[0]), wavelength=wavefront.get_wavelength(), copy_d=True)

    @classmethod
    def decoratePYNXWF(self, pynxwf):
        wavefront = PYNXWavefront(d=pynxwf.d, z=pynxwf.z, pixel_size=pynxwf.pixel_size, wavelength=pynxwf.wavelength, copy_d=True )

        #TODO check correctness
        #wavefront._xxx = 0.0

        return wavefront

#
# TESTS
#
def create_wavefront_generic(size_factor=1,pixel_size=1e-6,wavelength=1.5e-10):

    w = GenericWavefront2D.initialize_wavefront_from_steps(x_start=-0.5*pixel_size*512*size_factor,x_step=pixel_size,
                                                           y_start=-0.5*pixel_size*512*size_factor,y_step=pixel_size,
                                                           number_of_points=(512*size_factor,512*size_factor),wavelength=wavelength)
    w.set_plane_wave_from_complex_amplitude(complex_amplitude=(1.0+0.0j))
    w.clip_square(x_min=-100e-6,x_max=100e-6,y_min=-10e-6,y_max=10e-6)
    return w

def create_wavefront_pynx(size_factor=1,pixel_size=1e-6,wavelength=1.5e-10):
    # Near field propagation of a simple 20x200 microns slit

    w = Wavefront(d=numpy.zeros((512*size_factor, 512*size_factor), dtype=numpy.complex64), pixel_size=pixel_size, wavelength=wavelength)
    a = 20e-6 / 2
    x, y = w.get_x_y()
    print(x.min(),x.max(),y.min(),y.max())
    w.d = ((abs(y) < a) * (abs(x) < 100e-6)).astype(numpy.complex64)
    return w

def plot_wavefront_generic(w,show=True,title=None):
    z = w.get_intensity()
    x = w.get_coordinate_x()
    y = w.get_coordinate_y()

    if title is None:
        title="WOFRY"
    plot_image(z,1e6*x,1e6*y,title=title,xtitle='x [um]',ytitle='y [um]',show=show)

def plot_wavefront_pynx(w,do_shift=True,show=True,title=None):

    x, y = w.get_x_y()

    if do_shift:
        z = abs(numpy.fft.fftshift(w.d)).T
        # added srio
        z = z**2
    else:
        z = abs(w.d).T
    # z = abs((w.d)).T
    if title is None:
        title="Near field propagation (0.5m) of a 20x200 microns aperture"

    plot_image(z,
               1e6*numpy.linspace(x.min(),x.max(),num=z.shape[0],endpoint=True),
               1e6*numpy.linspace(y.min(),y.max(),num=z.shape[1],endpoint=True),
               title=title,
               xtitle='X (µm)',ytitle='Y (µm)',show=show)


if __name__ == "__main__":
    from srxraylib.plot.gol import plot_image

    #
    # GENERIC -> PYNX
    #
    w_generic = create_wavefront_generic()
    print(w_generic.get_complex_amplitude().shape)
    plot_wavefront_generic(w_generic,title='generic')

    w_pynx = PYNXWavefront.fromGenericWavefront(w_generic)
    print(">>>",w_pynx.d.shape)
    plot_wavefront_pynx(w_pynx,title="pynx from generic")
    w_pynx_d = PYNXWavefront.decoratePYNXWF(w_pynx)
    print(">>>",w_pynx_d.d.shape)
    plot_wavefront_pynx(w_pynx_d,title='pynx from generic decorated')

    #
    # PYNX ->  GENERIC
    #
    w_pynx = create_wavefront_pynx()
    plot_wavefront_pynx(w_pynx,title="pynx")
    w_pynx_d = PYNXWavefront.decoratePYNXWF(w_pynx)
    print(">>>",w_pynx_d.d.shape)
    plot_wavefront_pynx(w_pynx_d,title='pynx decorated')

    w_generic = w_pynx_d.toGenericWavefront()
    print(w_generic.get_complex_amplitude().shape)
    plot_wavefront_generic(w_generic,title='generic from pynx')





