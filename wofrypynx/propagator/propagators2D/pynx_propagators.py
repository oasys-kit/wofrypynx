

from wofry.propagator.propagator import Propagator2D
from wofrypynx.propagator.wavefront2D.pynx_wavefront import PYNXWavefront
from pynx.wavefront.cl_operator import FromPU, FreeFromPU, ToPU, PropagateNearField, PropagateFarField, PropagateFRT, default_processing_unit_wavefront


class FresnelPYNX(Propagator2D):

    HANDLER_NAME = "FRESNEL_PYNX"

    def get_handler_name(self):
        return self.HANDLER_NAME

    """
    2D Fresnel propagator using pynx (PropagateNearField)
    :param wavefront:
    :param propagation_distance:
    :return:
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters):

        is_generic_wavefront = isinstance(wavefront, GenericWavefront2D)

        if is_generic_wavefront:
            wavefront = PYNXWavefront.fromGenericWavefront(wavefront)
        else:
            if not isinstance(wavefront, PYNXWavefront): raise ValueError("wavefront cannot be managed by this propagator")

        #
        # propagation
        #
        wavefront_propagated = FromPU() * PropagateNearField(propagation_distance) * wavefront
        wavefront_propagated = PYNXWavefront.decoratePYNXWF(wavefront_propagated)

        if is_generic_wavefront:
            return wavefront_propagated.toGenericWavefront()
        else:
            return wavefront_propagated


class FraunhoferPYNX(Propagator2D):

    HANDLER_NAME = "FRAUNHOFER_PYNX"

    def get_handler_name(self):
        return self.HANDLER_NAME

    """
    2D Fresnel propagator using pynx (PropagateFarField)
    :param wavefront:
    :param propagation_distance:
    :return:
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters):

        is_generic_wavefront = isinstance(wavefront, GenericWavefront2D)

        if is_generic_wavefront:
            wavefront = PYNXWavefront.fromGenericWavefront(wavefront)
        else:
            if not isinstance(wavefront, PYNXWavefront): raise ValueError("wavefront cannot be managed by this propagator")

        #
        # propagation
        #
        wavefront_propagated = FromPU() * PropagateFarField(propagation_distance) * wavefront
        wavefront_propagated = PYNXWavefront.decoratePYNXWF(wavefront_propagated)

        if is_generic_wavefront:
            return wavefront_propagated.toGenericWavefront()
        else:
            return wavefront_propagated

class FrtPYNX(Propagator2D):

    HANDLER_NAME = "FRT_PYNX"

    def get_handler_name(self):
        return self.HANDLER_NAME

    """
    2D Fresnel propagator using pynx (PropagateFRT)
    :param wavefront:
    :param propagation_distance:
    :return:
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters):

        is_generic_wavefront = isinstance(wavefront, GenericWavefront2D)

        if is_generic_wavefront:
            wavefront = PYNXWavefront.fromGenericWavefront(wavefront)
        else:
            if not isinstance(wavefront, PYNXWavefront): raise ValueError("wavefront cannot be managed by this propagator")

        #
        # propagation
        #
        wavefront_propagated = FromPU() * PropagateFRT(propagation_distance) * wavefront
        wavefront_propagated = PYNXWavefront.decoratePYNXWF(wavefront_propagated)

        if is_generic_wavefront:
            return wavefront_propagated.toGenericWavefront()
        else:
            return wavefront_propagated


# todo: move outside
#
# test (main)
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

    w = PYNXWavefront(d=numpy.zeros((512*size_factor, 512*size_factor), dtype=numpy.complex64), pixel_size=pixel_size, wavelength=wavelength)
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

def propagate_wavefront(wf,propagation_distance,method='srw'):


        propagation_elements = PropagationElements()

        screen = WOScreen(name="Screen")

        propagation_elements.add_beamline_element(BeamlineElement(optical_element=screen,
                                                                  coordinates=ElementCoordinates(p=0, q=propagation_distance)))



        # initialize_default_propagator_2D()

        propagation_parameters = PropagationParameters(wavefront=wf,
                                                       propagation_elements=propagation_elements)

        if method == 'fresnel':

            propagation_parameters.set_additional_parameters("shift_half_pixel", True)
            wf1 = propagator.do_propagation(propagation_parameters, Fresnel2D.HANDLER_NAME)
        elif method == 'srw':
            propagation_parameters.set_additional_parameters("srw_autosetting", 0)
            wf1 = propagator.do_propagation(propagation_parameters, FresnelSRW.HANDLER_NAME)
        elif method == 'pynx':
            wf1 = propagator.do_propagation(propagation_parameters, FresnelPYNX.HANDLER_NAME)
        elif method == 'pynx2':
            wf1 = propagator.do_propagation(propagation_parameters, FraunhoferPYNX.HANDLER_NAME)
        elif method == 'pynx3':
            wf1 = propagator.do_propagation(propagation_parameters, FrtPYNX.HANDLER_NAME)
        else:
            raise Exception("Not implemented method: %s"%method)

        return wf1

if __name__ == "__main__":
    from srxraylib.plot.gol import plot_image

    import numpy
    import scipy.constants as codata
    angstroms_to_eV = codata.h*codata.c/codata.e*1e10
    from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
    from wofry.elements.optical_elements.ideal_elements.screen import WOScreen
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagator import PropagationElements


    from wofry.propagator.propagators2D.fresnel import Fresnel2D,FresnelConvolution2D
    from wofry.propagator.propagators2D.integral import Integral2D
    from wofrysrw.propagator.propagators2D.srw_fresnel import FresnelSRW

    from wofry.propagator.propagator import PropagationManager, PropagationParameters

    propagator = PropagationManager.Instance()
    propagator.add_propagator(Fresnel2D())
    propagator.add_propagator(FresnelPYNX())
    propagator.add_propagator(FraunhoferPYNX())
    propagator.add_propagator(FrtPYNX())
    propagator.add_propagator(FresnelSRW())

    #
    # GENERIC
    #
    # w_generic = create_wavefront_generic()
    # plot_wavefront_generic(w_generic,title='generic')
    #
    # w_generic_propagated = propagate_wavefront(w_generic,propagation_distance=0.5,method='pynx')
    # plot_wavefront_generic(w_generic_propagated,title='generic propagated pynx')
    #
    # w_generic_propagated = propagate_wavefront(w_generic,propagation_distance=0.5,method='fresnel')
    # plot_wavefront_generic(w_generic_propagated,title='generic propagated fresnel')
    #
    # w_generic_propagated = propagate_wavefront(w_generic,propagation_distance=0.5,method='srw')
    # plot_wavefront_generic(w_generic_propagated,title='generic propagated srw')

    #
    # PYNX
    #

    w_pynx_d = create_wavefront_pynx()
    print("Wavefront type: ",type(w_pynx_d))
    plot_wavefront_pynx(w_pynx_d,title='pynx decorated')
    w_pynx_d_propagated = propagate_wavefront(w_pynx_d,propagation_distance=0.5,method='pynx')
    plot_wavefront_pynx(w_pynx_d_propagated,title='pynx propagated pynx')

    # w_pynx_d2 = create_wavefront_pynx()
    # w_pynx_d_propagated2 = propagate_wavefront(w_pynx_d2,propagation_distance=0.5,method='fresnel')
    # print("Wavefront type PROP: ",type(w_pynx_d_propagated2))
    # plot_wavefront_pynx(w_pynx_d_propagated2,title='pynx propagated fresnel')


    # TODO: this should be possible, if the SRW propagator converts any wavefron to generic and back!
    # w_pynx_d3 = create_wavefront_pynx()
    # w_pynx_d_propagated3 = propagate_wavefront(w_pynx_d3,propagation_distance=0.5,method='srw')
    # print("Wavefront type: ",type(w_pynx_d_propagated3))
    # plot_wavefront_pynx(w_pynx_d_propagated3,title='pynx propagated srw')

