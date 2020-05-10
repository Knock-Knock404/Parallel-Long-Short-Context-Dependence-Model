import re
f = open("test.tsv","r")
s = 0
c = 0
ls = []
lenth = 0
while True:
	try:
		l = f.readline()
		if re.match(r"\s",l[0]):
			c += 1
			lenth += s
			ls.append(s)
			s = 0
			print(c)
			continue
		s += 1
	except Exception:
		break		
f.close() 

print(max(ls))
print(lenth/len(ls))
