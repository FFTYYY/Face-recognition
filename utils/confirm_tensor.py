def tensor_feature(t):
	res = t.sum()

	try: res += t[1][2].sum()
	except Exception: pass

	try: res += (t[2][1] * t[3][0]).sum()
	except Exception: pass

	try: res += (t < 0).float().mean()
	except Exception: pass

	try: res += (t[-1] < 0.5).float().mean()
	except Exception: pass

	return res
