function [illuminant] = get_illuminant(input_image,mask_chart)

patch_r = input_image(:,:,1);
patch_g = input_image(:,:,2);
patch_b = input_image(:,:,3);
patch_r = patch_r(mask_chart);
patch_g = patch_g(mask_chart);
patch_b = patch_b(mask_chart);

illuminant = [mean(patch_r), mean(patch_g), mean(patch_b)];

mag_illuminant = norm(illuminant);
illuminant(1) = illuminant(1)/mag_illuminant;
illuminant(2) = illuminant(2)/mag_illuminant;
illuminant(3) = illuminant(3)/mag_illuminant;
end