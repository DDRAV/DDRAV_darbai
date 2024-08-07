PGDMP      8                |           Miniprojektas2    16.3    16.3 $    �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            �           1262    16814    Miniprojektas2    DATABASE     �   CREATE DATABASE "Miniprojektas2" WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_United States.1252';
     DROP DATABASE "Miniprojektas2";
                postgres    false            �            1259    16824    Darbuotuojas    TABLE     �   CREATE TABLE public."Darbuotuojas" (
    "ID" integer NOT NULL,
    vardas character varying,
    pavarde character varying,
    darbuotuojonumeris integer,
    darbuotuojopastas character varying,
    gimimodata date,
    alga integer
);
 "   DROP TABLE public."Darbuotuojas";
       public         heap    postgres    false            �            1259    16823    Darbuotuojas_ID_seq    SEQUENCE     �   ALTER TABLE public."Darbuotuojas" ALTER COLUMN "ID" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public."Darbuotuojas_ID_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
            public          postgres    false    218            �            1259    16832    Klientas    TABLE     �   CREATE TABLE public."Klientas" (
    "ID" integer NOT NULL,
    pavadinimas character varying,
    klientopastas character varying,
    klientonumeris integer,
    statusas character varying,
    vadybininkoid integer
);
    DROP TABLE public."Klientas";
       public         heap    postgres    false            �            1259    16831    Klientas_ID_seq    SEQUENCE     �   ALTER TABLE public."Klientas" ALTER COLUMN "ID" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public."Klientas_ID_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
            public          postgres    false    220            �            1259    16845    Sandelys    TABLE     �   CREATE TABLE public."Sandelys" (
    "ID" integer NOT NULL,
    klientoid integer,
    salis character varying,
    pastokodas character varying,
    adresas character varying,
    darbolaikas tsrange,
    sandeliotipas character varying
);
    DROP TABLE public."Sandelys";
       public         heap    postgres    false            �            1259    16844    Sandelys_ID_seq    SEQUENCE     �   ALTER TABLE public."Sandelys" ALTER COLUMN "ID" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public."Sandelys_ID_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
            public          postgres    false    222            �            1259    16858 	   Uzsakymas    TABLE     3  CREATE TABLE public."Uzsakymas" (
    "ID" integer NOT NULL,
    frachtas integer,
    pakrovimosandelis integer,
    pakrovimolaikas time without time zone,
    iskrovimosandelis integer,
    iskrovimolaikas time without time zone,
    vadybininkoid integer,
    vezejoid integer,
    klientoid integer
);
    DROP TABLE public."Uzsakymas";
       public         heap    postgres    false            �            1259    16857    Uzsakymas_ID_seq    SEQUENCE     �   ALTER TABLE public."Uzsakymas" ALTER COLUMN "ID" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public."Uzsakymas_ID_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
            public          postgres    false    224            �            1259    16816    Vezejas    TABLE     {   CREATE TABLE public."Vezejas" (
    "ID" integer NOT NULL,
    pavadinimas character varying,
    vezejonumeris integer
);
    DROP TABLE public."Vezejas";
       public         heap    postgres    false            �            1259    16815    Vezejas_ID_seq    SEQUENCE     �   ALTER TABLE public."Vezejas" ALTER COLUMN "ID" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public."Vezejas_ID_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
            public          postgres    false    216            �          0    16824    Darbuotuojas 
   TABLE DATA           x   COPY public."Darbuotuojas" ("ID", vardas, pavarde, darbuotuojonumeris, darbuotuojopastas, gimimodata, alga) FROM stdin;
    public          postgres    false    218   �+       �          0    16832    Klientas 
   TABLE DATA           o   COPY public."Klientas" ("ID", pavadinimas, klientopastas, klientonumeris, statusas, vadybininkoid) FROM stdin;
    public          postgres    false    220   �+       �          0    16845    Sandelys 
   TABLE DATA           m   COPY public."Sandelys" ("ID", klientoid, salis, pastokodas, adresas, darbolaikas, sandeliotipas) FROM stdin;
    public          postgres    false    222   �+       �          0    16858 	   Uzsakymas 
   TABLE DATA           �   COPY public."Uzsakymas" ("ID", frachtas, pakrovimosandelis, pakrovimolaikas, iskrovimosandelis, iskrovimolaikas, vadybininkoid, vezejoid, klientoid) FROM stdin;
    public          postgres    false    224   �+       �          0    16816    Vezejas 
   TABLE DATA           E   COPY public."Vezejas" ("ID", pavadinimas, vezejonumeris) FROM stdin;
    public          postgres    false    216   ,       �           0    0    Darbuotuojas_ID_seq    SEQUENCE SET     D   SELECT pg_catalog.setval('public."Darbuotuojas_ID_seq"', 1, false);
          public          postgres    false    217            �           0    0    Klientas_ID_seq    SEQUENCE SET     @   SELECT pg_catalog.setval('public."Klientas_ID_seq"', 1, false);
          public          postgres    false    219            �           0    0    Sandelys_ID_seq    SEQUENCE SET     @   SELECT pg_catalog.setval('public."Sandelys_ID_seq"', 1, false);
          public          postgres    false    221            �           0    0    Uzsakymas_ID_seq    SEQUENCE SET     A   SELECT pg_catalog.setval('public."Uzsakymas_ID_seq"', 1, false);
          public          postgres    false    223            �           0    0    Vezejas_ID_seq    SEQUENCE SET     ?   SELECT pg_catalog.setval('public."Vezejas_ID_seq"', 1, false);
          public          postgres    false    215            1           2606    16830    Darbuotuojas Darbuotuojas_pkey 
   CONSTRAINT     b   ALTER TABLE ONLY public."Darbuotuojas"
    ADD CONSTRAINT "Darbuotuojas_pkey" PRIMARY KEY ("ID");
 L   ALTER TABLE ONLY public."Darbuotuojas" DROP CONSTRAINT "Darbuotuojas_pkey";
       public            postgres    false    218            3           2606    16838    Klientas Klientas_pkey 
   CONSTRAINT     Z   ALTER TABLE ONLY public."Klientas"
    ADD CONSTRAINT "Klientas_pkey" PRIMARY KEY ("ID");
 D   ALTER TABLE ONLY public."Klientas" DROP CONSTRAINT "Klientas_pkey";
       public            postgres    false    220            5           2606    16851    Sandelys Sandelys_pkey 
   CONSTRAINT     Z   ALTER TABLE ONLY public."Sandelys"
    ADD CONSTRAINT "Sandelys_pkey" PRIMARY KEY ("ID");
 D   ALTER TABLE ONLY public."Sandelys" DROP CONSTRAINT "Sandelys_pkey";
       public            postgres    false    222            7           2606    16862    Uzsakymas Uzsakymas_pkey 
   CONSTRAINT     \   ALTER TABLE ONLY public."Uzsakymas"
    ADD CONSTRAINT "Uzsakymas_pkey" PRIMARY KEY ("ID");
 F   ALTER TABLE ONLY public."Uzsakymas" DROP CONSTRAINT "Uzsakymas_pkey";
       public            postgres    false    224            /           2606    16822    Vezejas Vezejas_pkey 
   CONSTRAINT     X   ALTER TABLE ONLY public."Vezejas"
    ADD CONSTRAINT "Vezejas_pkey" PRIMARY KEY ("ID");
 B   ALTER TABLE ONLY public."Vezejas" DROP CONSTRAINT "Vezejas_pkey";
       public            postgres    false    216            :           2606    16868    Uzsakymas iskrovimosandelis 
   FK CONSTRAINT     �   ALTER TABLE ONLY public."Uzsakymas"
    ADD CONSTRAINT iskrovimosandelis FOREIGN KEY (iskrovimosandelis) REFERENCES public."Sandelys"("ID");
 G   ALTER TABLE ONLY public."Uzsakymas" DROP CONSTRAINT iskrovimosandelis;
       public          postgres    false    222    4661    224            9           2606    16852    Sandelys klientoid 
   FK CONSTRAINT     |   ALTER TABLE ONLY public."Sandelys"
    ADD CONSTRAINT klientoid FOREIGN KEY (klientoid) REFERENCES public."Klientas"("ID");
 >   ALTER TABLE ONLY public."Sandelys" DROP CONSTRAINT klientoid;
       public          postgres    false    222    220    4659            ;           2606    16883    Uzsakymas klientoid 
   FK CONSTRAINT     }   ALTER TABLE ONLY public."Uzsakymas"
    ADD CONSTRAINT klientoid FOREIGN KEY (klientoid) REFERENCES public."Klientas"("ID");
 ?   ALTER TABLE ONLY public."Uzsakymas" DROP CONSTRAINT klientoid;
       public          postgres    false    220    4659    224            <           2606    16863    Uzsakymas pakrovimosandelis 
   FK CONSTRAINT     �   ALTER TABLE ONLY public."Uzsakymas"
    ADD CONSTRAINT pakrovimosandelis FOREIGN KEY (pakrovimosandelis) REFERENCES public."Sandelys"("ID");
 G   ALTER TABLE ONLY public."Uzsakymas" DROP CONSTRAINT pakrovimosandelis;
       public          postgres    false    4661    224    222            =           2606    16873    Uzsakymas vadybinikoid 
   FK CONSTRAINT     �   ALTER TABLE ONLY public."Uzsakymas"
    ADD CONSTRAINT vadybinikoid FOREIGN KEY (vadybininkoid) REFERENCES public."Darbuotuojas"("ID");
 B   ALTER TABLE ONLY public."Uzsakymas" DROP CONSTRAINT vadybinikoid;
       public          postgres    false    224    218    4657            8           2606    16839    Klientas vadybininkoid 
   FK CONSTRAINT     �   ALTER TABLE ONLY public."Klientas"
    ADD CONSTRAINT vadybininkoid FOREIGN KEY (vadybininkoid) REFERENCES public."Darbuotuojas"("ID");
 B   ALTER TABLE ONLY public."Klientas" DROP CONSTRAINT vadybininkoid;
       public          postgres    false    218    4657    220            >           2606    16878    Uzsakymas vezejoid 
   FK CONSTRAINT     z   ALTER TABLE ONLY public."Uzsakymas"
    ADD CONSTRAINT vezejoid FOREIGN KEY (vezejoid) REFERENCES public."Vezejas"("ID");
 >   ALTER TABLE ONLY public."Uzsakymas" DROP CONSTRAINT vezejoid;
       public          postgres    false    216    224    4655            �   
   x������ � �      �   
   x������ � �      �   
   x������ � �      �   
   x������ � �      �   
   x������ � �     