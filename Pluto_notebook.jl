### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ e3bca987-b7dd-44aa-8170-928977a7f9d4
using Pkg

# ╔═╡ 16d7eb90-9b8f-11eb-24e3-079cd0b2159a
using MAT

# ╔═╡ ca1b2092-85b1-43db-8843-daef4e9fd4e6
using GaussianProcesses

# ╔═╡ 51576eba-22b6-4b2b-a5f8-d46a683d0634
using DSP

# ╔═╡ e058aa81-844d-4bb3-949e-66da307b6715
using ImageFiltering

# ╔═╡ f4feba3d-53c3-4f0b-83a4-18bde59f4bf2
using Statistics

# ╔═╡ 2ff3ae2a-b976-4858-a0e4-a7a1ca658953
using Plots

# ╔═╡ a857a483-0d58-4e9d-9baa-bb76061c8d31
using StatsBase

# ╔═╡ 6c2ee5ce-283b-48ed-9e8b-d856060dec95
md"# Gaussian Process Hippocampus #"

# ╔═╡ 481ea5ef-0902-4b44-b58f-e38c2d989eae
md"### Packages ###"

# ╔═╡ 2930743f-d472-4592-a52b-0fac6389ca2b


# ╔═╡ a1c5d767-3981-4d2e-84a1-45408794965d
#Pkg.add("DSP")

# ╔═╡ 0dd70a9f-2b26-444a-82fc-0dc5fd2edc11
md"### Place cell map ###"

# ╔═╡ 983e91e5-3eba-4006-8da3-2662b954de58
nbinshist = 50; 

# ╔═╡ 7c03fd98-4fde-49a5-8841-409c6707475e
smwinhist = 10;

# ╔═╡ ae87962e-98f5-4b4d-898c-6b04a30a8524
begin
	smkernel = DSP.Windows.hanning(smwinhist)*DSP.Windows.hanning(smwinhist)';
	smkernel = smkernel./sum(smkernel[:]);
end

# ╔═╡ c34a44ea-3cb5-445a-bd53-a5a78d6de974
md"### GP model ((x,y)->S) ###"

# ╔═╡ 7b38ad08-1624-4a94-a64d-6fb694d270da
kern = Mat52Iso(4., 0.5) # taken from a particular fit

# ╔═╡ b8ae7682-1920-46b7-ac5c-81bf9f6a5307
nbins = 15; 

# ╔═╡ dfb2c9b3-d4c9-4c05-a0f1-fb65af4ee437
begin
	Xu = zeros(nbins*nbins,2);
	for i = 1:nbins
		for j = 1:nbins
			Xu[(i-1)*nbins + j,1] = i*450/nbins; 
			Xu[(i-1)*nbins + j,2] = j*450/nbins;
		end
	end
end

# ╔═╡ 9c26a1df-f552-4f23-b33d-7b3dbbfed747
md"### Loading data ###"

# ╔═╡ f467717b-feb7-43e2-a18d-90de8fd53218
filename = joinpath("data", "LMN73101219.mat")

# ╔═╡ c5feec25-3b15-4f63-b6e2-fc54062a08ed
vars = matread(filename);

# ╔═╡ 82ccb20d-9cfe-4227-a22e-298aa9472c3c
X= vars["XDLC"]["Head"];

# ╔═╡ a6bfd49e-3ef9-4902-9c8a-79e3385de934
Y = vars["YDLC"]["Head"];

# ╔═╡ 48c708bf-75f4-4f32-a677-24d71da553cc
h_xy = fit(Histogram, (vec(Y),vec(X)),nbins=nbinshist);

# ╔═╡ e212e9a1-52b8-4033-a3ff-1af9c7d6dd5a
smocc = imfilter(h_xy.weights, smkernel);

# ╔═╡ 3527c149-6ce8-4e47-9a5d-ef401ec72834
maskhist = smocc.>5;

# ╔═╡ ec3689c3-a035-4996-a605-061e198c6b87
#histogram2d(X, Y,bins=450,aspect_ratio=:equal)

# ╔═╡ fab87d09-4f61-4896-b648-ecf86c2dc059
md"### Plotting ###"

# ╔═╡ 106831d9-5104-46d2-b75a-8aa1b54474fc
neuroni = 10; 

# ╔═╡ 3a4a4688-5ba8-4609-af40-63f546c44dd1
h_s = fit(Histogram, (vec(Y), vec(X)),weights(vec(vars["S"][neuroni,:])'),nbins=nbinshist);

# ╔═╡ f7434f8f-5ebc-4687-acb6-29056b41b32a
place_map = maskhist.*(imfilter(h_s.weights, smkernel)./smocc); 

# ╔═╡ 3b06fac4-9e2a-4e91-9e76-d306bdd1d392
ind_toGP = findall(vars["S"][neuroni,:].>0);

# ╔═╡ 32c18537-9a70-4f45-adc0-7edada9e07f6
x_toGP = hcat(X,Y)[ind_toGP,:];

# ╔═╡ 67a9b40d-5288-4290-8184-2af6c3b84216
y_toGP = vars["S"][neuroni,:][ind_toGP];

# ╔═╡ ab583510-5df1-4877-8019-8b315380c508
mean_toGP = MeanConst(mean(y_toGP))               

# ╔═╡ e3a427ba-6b45-41a9-ae0a-085b41bd18c8
gp_SOR = GaussianProcesses.SoR(x_toGP', Xu', y_toGP, MeanConst(mean(y_toGP)), kern, log(std(y_toGP)));

# ╔═╡ 16b507fd-6ae0-47d5-a5b1-4a1f979d8722
begin
	tmpXs = zeros(nbinshist*nbinshist,2);
	for i = 1:nbinshist
		for j = 1:nbinshist
			tmpXs[(i-1)*nbinshist + j,1] = i*450/nbinshist; 
			tmpXs[(i-1)*nbinshist + j,2] = j*450/nbinshist;
		end
	end
	tmpYs = predict_y(gp_SOR, tmpXs'; full_cov=false);
	GPhist = reshape(tmpYs[1],nbinshist,nbinshist);
	GPhist;
end

# ╔═╡ 9acc189b-e5be-4de5-9fdf-a36536dc44b2
heatmap(place_map,aspect_ratio=:equal)

# ╔═╡ 3ab6e26b-29a0-4294-83e2-595ebd996e50
heatmap(GPhist, aspect_ratio=:equal)

# ╔═╡ c557108c-041e-4ad0-9bcf-2ddf23871a5a
#heatmap(gp_SOR,aspect_ratio=:equal)

# ╔═╡ 65a54364-42b5-4f1f-b128-b97ffaa767e4
#optimize!(gp_SOR)

# ╔═╡ Cell order:
# ╠═6c2ee5ce-283b-48ed-9e8b-d856060dec95
# ╠═481ea5ef-0902-4b44-b58f-e38c2d989eae
# ╠═2930743f-d472-4592-a52b-0fac6389ca2b
# ╠═16d7eb90-9b8f-11eb-24e3-079cd0b2159a
# ╠═ca1b2092-85b1-43db-8843-daef4e9fd4e6
# ╠═e3bca987-b7dd-44aa-8170-928977a7f9d4
# ╠═a1c5d767-3981-4d2e-84a1-45408794965d
# ╠═51576eba-22b6-4b2b-a5f8-d46a683d0634
# ╠═e058aa81-844d-4bb3-949e-66da307b6715
# ╠═f4feba3d-53c3-4f0b-83a4-18bde59f4bf2
# ╠═2ff3ae2a-b976-4858-a0e4-a7a1ca658953
# ╠═a857a483-0d58-4e9d-9baa-bb76061c8d31
# ╟─0dd70a9f-2b26-444a-82fc-0dc5fd2edc11
# ╠═983e91e5-3eba-4006-8da3-2662b954de58
# ╠═7c03fd98-4fde-49a5-8841-409c6707475e
# ╠═ae87962e-98f5-4b4d-898c-6b04a30a8524
# ╠═48c708bf-75f4-4f32-a677-24d71da553cc
# ╠═3a4a4688-5ba8-4609-af40-63f546c44dd1
# ╠═e212e9a1-52b8-4033-a3ff-1af9c7d6dd5a
# ╠═3527c149-6ce8-4e47-9a5d-ef401ec72834
# ╠═f7434f8f-5ebc-4687-acb6-29056b41b32a
# ╟─c34a44ea-3cb5-445a-bd53-a5a78d6de974
# ╠═32c18537-9a70-4f45-adc0-7edada9e07f6
# ╠═67a9b40d-5288-4290-8184-2af6c3b84216
# ╠═16b507fd-6ae0-47d5-a5b1-4a1f979d8722
# ╠═3b06fac4-9e2a-4e91-9e76-d306bdd1d392
# ╠═ab583510-5df1-4877-8019-8b315380c508
# ╠═7b38ad08-1624-4a94-a64d-6fb694d270da
# ╠═b8ae7682-1920-46b7-ac5c-81bf9f6a5307
# ╠═dfb2c9b3-d4c9-4c05-a0f1-fb65af4ee437
# ╠═e3a427ba-6b45-41a9-ae0a-085b41bd18c8
# ╟─9c26a1df-f552-4f23-b33d-7b3dbbfed747
# ╠═f467717b-feb7-43e2-a18d-90de8fd53218
# ╠═c5feec25-3b15-4f63-b6e2-fc54062a08ed
# ╠═82ccb20d-9cfe-4227-a22e-298aa9472c3c
# ╠═a6bfd49e-3ef9-4902-9c8a-79e3385de934
# ╠═ec3689c3-a035-4996-a605-061e198c6b87
# ╟─fab87d09-4f61-4896-b648-ecf86c2dc059
# ╠═106831d9-5104-46d2-b75a-8aa1b54474fc
# ╠═9acc189b-e5be-4de5-9fdf-a36536dc44b2
# ╠═3ab6e26b-29a0-4294-83e2-595ebd996e50
# ╠═c557108c-041e-4ad0-9bcf-2ddf23871a5a
# ╠═65a54364-42b5-4f1f-b128-b97ffaa767e4
