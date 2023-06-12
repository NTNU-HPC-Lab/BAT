
#include "dedispersion.h"

extern "C"
void dedispersion_reference(const unsigned char * input, float * output, const float * shifts) {
	for ( unsigned int dm = 0; dm < nr_dms; dm++ ) {
		for ( unsigned int sample = 0; sample < nr_samples; sample++ ) {
			float dedispersedSample = 0.0f;

			for ( unsigned int channel = 0; channel < nr_channels; channel++ ) {
				unsigned int shift = (dm_first + dm*dm_step) * shifts[channel];

				dedispersedSample += input[(channel * nr_samples_per_channel) + (sample + shift)];
			}

			output[(dm * nr_samples) + sample] = dedispersedSample;
		}
	}
}
